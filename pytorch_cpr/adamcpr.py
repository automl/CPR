# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import cast, List, Optional, Tuple, Union
import functools

import torch
from torch import Tensor
from torch.compiler import is_compiling
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _get_scalar_dtype,
    _get_value,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)
from .group_parameter import group_parameters_for_cpr_optimizer

###
# copy code snippets from PyTorch 2.5 to make AdamCPR compatible to PyTorch 2.3.1+
def _get_capturable_supported_devices(supports_xla: bool = True) -> List[str]:
    r"""Return the device type list that supports capturable optimizer."""
    capturable_supported_devices = ["cuda", "xpu", "hpu"]
    if not torch.jit.is_scripting():
        capturable_supported_devices.append(torch._C._get_privateuse1_backend_name())
    if supports_xla:
        capturable_supported_devices.append("xla")
    return capturable_supported_devices

def _disable_dynamo_if_unsupported(single_tensor_fn=None):
    # workaround for torchscript BC
    # it requires all called functions to be in the
    # global environment at the site at which the
    # maybe_fallback closure is created
    if single_tensor_fn:
        globals()[single_tensor_fn.__name__] = single_tensor_fn

    def wrapper(func):
        import inspect

        disabled_func = torch._disable_dynamo(func)
        ps = inspect.signature(func).parameters
        has_state_steps = True
        try:
            state_steps_ind = list(ps.keys()).index("state_steps")
        except ValueError:
            has_state_steps = False

        # Today, there are cases where we stack state steps
        # and pass them as the value arg of foreach ops.
        # Having state steps on cuda as the value arg is not supported in eager,
        # but this only occurs in the rare case that the user explicitly deletes
        # the capturable flag. If capturable=True, this is not a problem.
        @functools.wraps(func)
        def maybe_fallback(*args, **kwargs):
            if is_compiling() and (
                not kwargs.get("capturable", False)
                and has_state_steps
                and (args[state_steps_ind] and args[state_steps_ind][0].is_cuda)
                or (
                    "state_steps" in kwargs
                    and kwargs["state_steps"]
                    and kwargs["state_steps"][0].is_cuda
                )
            ):
                return disabled_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return maybe_fallback

    return wrapper


def single_initilize_kappa(kappa, param, reg_function):
    if reg_function == 'l2':
        kappa.mul_(0).add_(param.square().sum())
    elif reg_function == 'std':
        kappa.mul_(0).add_(param.std())
    elif reg_function == 'l1':
        kappa.mul_(0).add_(param.abs().sum())
    elif reg_function == 'huber':
        param_abs = param.abs()
        huber_loss = torch.where(param_abs < 1, 0.5 * param.square(), param_abs - 0.5)
        kappa.mul_(0).add_(huber_loss.sum())


__all__ = ["AdamCPR", "adamcpr"]
HIGHKAPPA=1e6


class AdamCPR(Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: Union[float, Tensor] = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            kappa_init_method: str = 'inflection_point',
            kappa_init_param: float = 1000,
            reg_function: str = 'l2',
            kappa_update: float = 1.0,
            reg_step_size: int = 200,
            reg_ema_decay: float = 0.99,
            reg_embedding: bool = False,
            reg_by_lr: bool = False,
            amsgrad: bool = False,
            *,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            capturable: bool = False,
            differentiable: bool = False,
    ):
        """
        Implements AdamCPR algorithm.

        Please find more details in the CPR paper: https://arxiv.org/abs/2311.09058

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
                is not yet supported for all our implementations. Please use a float
                LR if you are not also specifying capturable=True.
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            kappa_init_method (str, optional): Method to initilize the upper bound of the regularization term (default: inflection_point).
                Options are 'uniform', 'warm_start', 'dependent', 'inflection_point'.
            kappa_init_param (float, optional): The value to initialize the upper bound of the regularization term (default: 1000).
            reg_function (str, optional): The regularization function to use (default: 'l2').
                Options are 'l2', 'l1', 'std', 'huber'.
            kappa_update (float, optional): The update rate for the regularization term (default: 1.0).
            reg_step_size (int, optional): The sampling rate to detect the inflection point (default: 200).
            reg_ema_decay (float, optional): The decay rate for the exponential moving average of the inflection point (default: 0.9).
            reg_embedding (bool, optional): Whether to regularize the embedding layer (default: False).
            reg_by_lr (bool, optional): Whether to scale the regularization term by the learning rate (default: False).
            amsgrad (bool, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False)
        """
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        if not 0.0 <= kappa_update:
            raise ValueError(f"Invalid kappa_update value: {kappa_update}")

        self.reg_function = reg_function
        self.kappa_init_method = kappa_init_method
        self.reg_step_size = reg_step_size
        self.reg_ema_decay = reg_ema_decay
        self.reg_by_lr = reg_by_lr

        if self.kappa_init_method not in ['warm_start', 'uniform', 'dependent', 'inflection_point']:
            raise ValueError(f"Invalid kappa_init_method: {kappa_init_method}")
        if self.kappa_init_method == "warm_start":
            self.warm_start = kappa_init_param
        else:
            self.warm_start = 0
            self.kappa_init_param = kappa_init_param

        self.kappa_update = kappa_update

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            regularize=False,
            kappa_update=kappa_update,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
        )

        if isinstance(params, torch.nn.Module):
            params = group_parameters_for_cpr_optimizer(params, regularize_embedding=reg_embedding)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            lagmuls,
            kappas,
            kappa_updates,
            prev_regs,
            prev_reg_gradients,
            inflection_point_emas,
            state_steps,
    ):
        has_complex = False

        if "regularize" not in group:
            if "weight_decay" in group:
                if group["weight_decay"] != 0.0:
                    group['regularize'] = True
                else:
                    group['regularize'] = False

        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamCPR does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:

                state['regularize'] = group['regularize']

                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(),
                        device=p.device,
                    )
                    if group["capturable"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state['lagmul'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state['prev_reg'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state['prev_reg_gradient'] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state['inflection_point_emas'] = torch.tensor([0.0], dtype=torch.float, device=p.device)

                if self.reg_function == 'std':
                    state['kappa_update'] = torch.tensor(self.kappa_update, dtype=torch.float, device=p.device)
                else:
                    state['kappa_update'] = torch.tensor(self.kappa_update / p.numel(), dtype=torch.float,
                                                         device=p.device)

                if self.kappa_init_method == 'uniform':
                    state["kappa"] = torch.tensor(self.kappa_init_param, dtype=torch.float, device=p.device)
                elif self.kappa_init_method == 'warm_start':
                    state["kappa"] = torch.tensor(0.0, dtype=torch.float, device=p.device)
                elif self.kappa_init_method == 'inflection_point':
                    # Initialize kappa with HIGHKAPPA, it doesn't have any effect to the training before the inflection
                    # point and HIGHKAPPA will be used to identify un-set kappa bounds
                    state["kappa"] = torch.tensor(HIGHKAPPA, dtype=torch.float, device=p.device)
                elif self.kappa_init_method == 'dependent':
                    kappa = torch.tensor(0.0, dtype=torch.float, device=p.device)
                    single_initilize_kappa(kappa, p, self.reg_function)
                    state["kappa"] = self.kappa_init_param * kappa.detach()

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            lagmuls.append(state['lagmul'])
            kappas.append(state['kappa'])
            kappa_updates.append(state['kappa_update'])
            prev_regs.append(state['prev_reg'])
            prev_reg_gradients.append(state['prev_reg_gradient'])
            inflection_point_emas.append(state['inflection_point_emas'])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                    group["foreach"]
                    and isinstance(group["lr"], Tensor)
                    and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )

            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            lagmuls: List[Tensor] = []
            kappas: List[Tensor] = []
            kappa_updates: List[Tensor] = []
            prev_regs: List[Tensor] = []
            prev_reg_gradients: List[Tensor] = []
            inflection_point_emas: List[Tensor] = []
            state_steps: List[Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lagmuls,
                kappas,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                inflection_point_emas,
                state_steps,
            )

            adamcpr(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lagmuls,
                kappas,
                kappa_updates,
                prev_regs,
                prev_reg_gradients,
                inflection_point_emas,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                regularize=group['regularize'],
                warm_start=self.warm_start,
                reg_function=self.reg_function,
                reg_by_lr=self.reg_by_lr,
                kappa_init_method=self.kappa_init_method,
                reg_step_size=self.reg_step_size,
                reg_ema_decay=self.reg_ema_decay,
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return loss


@torch.jit.script
def l2_update(param, lagmul, kappa, kappa_update, reg_by_lr: bool, lr: float):
    sum_l2norm = param.square().sum()
    constraint_value = sum_l2norm - kappa
    lagmul.add_(kappa_update * constraint_value).clip_(min=0.)
    if reg_by_lr:
        param.addcmul_(param, lagmul, value=-2 * lr)
    else:
        param.addcmul_(param, lagmul, value=-2)


@torch.jit.script
def l1_update(param, lagmul, kappa, kappa_update, reg_by_lr: bool, lr: float):
    sum_l1norm = param.abs().sum()
    constraint_value = sum_l1norm - kappa
    lagmul.add_(kappa_update * constraint_value).clip_(min=0.)
    if reg_by_lr:
        param.addcmul_(param.sign(), lagmul, value=-lr)
    else:
        param.addcmul_(param.sign(), lagmul, value=-1)


@torch.jit.script
def std_update(param, lagmul, kappa, kappa_update, reg_by_lr: bool, lr: float):
    n = param.numel()
    std_dev = param.std()
    constraint_value = std_dev - kappa
    mean = param.mean()
    norm_param = param.sub(mean)
    grad_std_dev = norm_param.mul_(2).sub_(2 * norm_param.mean()).div_(n - 1)
    grad_std_dev.div_(std_dev.mul_(2))

    lagmul.add_(kappa_update * constraint_value).clip_(min=0.)
    if reg_by_lr:
        param.addcmul_(grad_std_dev, lagmul, value=-lr)
    else:
        param.addcmul_(grad_std_dev, lagmul, value=-1)


@torch.jit.script
def huber_update(param, lagmul, kappa, kappa_update, reg_by_lr: bool, lr: float):
    param_abs = param.abs()
    huber_idx = param_abs < 1
    huber_loss = torch.where(huber_idx, 0.5 * param.square(), param_abs - 0.5)
    sum_huber_loss = huber_loss.sum()
    constraint_value = sum_huber_loss - kappa
    lagmul.add_(kappa_update * constraint_value).clip_(min=0.)
    grad_huber = torch.where(huber_idx, param, param.sign())
    if reg_by_lr:
        param.addcmul_(grad_huber, lagmul, value=-lr)
    else:
        param.addcmul_(grad_huber, lagmul, value=-1)


def _single_tensor_adamcpr(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        lagmuls: List[Tensor],
        kappas: List[Tensor],
        kappa_updates: List[Tensor],
        prev_regs: List[Tensor],
        prev_reg_gradients: List[Tensor],
        inflection_point_emas: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        regularize: bool,
        warm_start: int,
        reg_function: str,
        reg_by_lr: bool,
        kappa_init_method: str,
        reg_step_size: int,
        reg_ema_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        lagmul = lagmuls[i]
        kappa = kappas[i]
        kappa_update = kappa_updates[i]
        prev_reg = prev_regs[i]
        prev_reg_gradient = prev_reg_gradients[i]
        inflection_point_ema = inflection_point_emas[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                    param.device.type == step_t.device.type
                    and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                        max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                        exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2 ** 0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        if regularize:
            if kappa_init_method == 'inflection_point' and kappa == 1000:
                current_l2m = param.square().sum()
                inflection_point_ema.mul_(reg_ema_decay).add_(current_l2m, alpha=1 - reg_ema_decay)
                if step > reg_step_size * 1 and step % reg_step_size == 0:
                    current_reg_gradient = inflection_point_ema - prev_reg
                    # Peak detection for gradient
                    if step > reg_step_size * 3 and prev_reg_gradient > current_reg_gradient:
                        single_initilize_kappa(kappa, param, reg_function)
                    # Update previous values for next iteration
                    prev_reg.copy_(inflection_point_ema)
                    if step > reg_step_size * 2:
                        prev_reg_gradient.copy_(current_reg_gradient)

            elif step > warm_start:
                if reg_function == 'l2':
                    l2_update(param, lagmul, kappa, kappa_update, reg_by_lr, lr)
                elif reg_function == 'std':
                    std_update(param, lagmul, kappa, kappa_update, reg_by_lr, lr)
                elif reg_function == 'l1':
                    l1_update(param, lagmul, kappa, kappa_update, reg_by_lr, lr)
                elif reg_function == 'huber':
                    huber_update(param, lagmul, kappa, kappa_update, reg_by_lr, lr)
                else:
                    raise ValueError(f"Unsupported regularization function: {reg_function}")

            elif kappa_init_method == 'warm_start' and step == warm_start:
                single_initilize_kappa(kappa, param, reg_function)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adamcpr(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        lagmuls: List[Tensor],
        kappas: List[Tensor],
        kappa_updates: List[Tensor],
        prev_regs: List[Tensor],
        prev_reg_gradients: List[Tensor],
        inflection_point_emas: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        regularize: bool,
        warm_start: int,
        reg_function: str,
        reg_by_lr: bool,
        kappa_init_method: str,
        reg_step_size: int,
        reg_ema_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, lagmuls, kappas, kappa_updates, prev_regs,
         prev_reg_gradients, inflection_point_emas, state_steps]
    )
    for (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs_,
            device_lagmuls_,
            device_kappas_,
            device_kappa_updates_,
            device_prev_regs_,
            device_prev_reg_gradients_,
            device_inflection_point_emas_,
            device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_lagmuls = cast(List[Tensor], device_lagmuls_)
        device_kappas = cast(List[Tensor], device_kappas_)
        device_kappa_updates = cast(List[Tensor], device_kappa_updates_)
        device_prev_regs = cast(List[Tensor], device_prev_regs_)
        device_prev_reg_gradients = cast(List[Tensor], device_prev_reg_gradients_)
        device_inflection_point_emas = cast(List[Tensor], device_inflection_point_emas_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                    device_lagmuls,
                    device_kappas,
                    device_kappa_updates,
                    device_prev_regs,
                    device_prev_reg_gradients,
                    device_inflection_point_emas,
                )
            else:
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_lagmuls,
                    device_kappas,
                    device_kappa_updates,
                    device_prev_regs,
                    device_prev_reg_gradients,
                    device_inflection_point_emas,
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
        )

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [
                bc ** 0.5 for bc in bias_correction2  # type: ignore[arg-type]
            ]

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(
                device_params,
                device_exp_avgs,
                exp_avg_sq_sqrt,
                step_size,  # type: ignore[arg-type]
            )

        if regularize:
            if device_state_steps[0] > warm_start:
                if reg_function == 'l2':
                    square_sum_params = torch._foreach_pow(torch._foreach_norm(device_params), 2)
                    torch._foreach_sub_(square_sum_params, device_kappas)
                    torch._foreach_mul_(square_sum_params, device_kappa_updates)
                    torch._foreach_add_(device_lagmuls, square_sum_params)
                    for lagmul in device_lagmuls:
                        lagmul.clip_(min=0.)
                    if reg_by_lr:
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -2 * lr)
                    else:
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -2)

                elif reg_function == 'std':
                    std_params, ns = [], []
                    for device_param in device_params:
                        std_params.append(device_param.std().unsqueeze(0))
                        ns.append(device_param.numel() - 1)
                    mean_params = [device_param.mean() for device_param in device_params]
                    norm_params = torch._foreach_sub(device_params, mean_params)
                    mean_norm_params = [norm_param.mean() * 2 for norm_param in norm_params]

                    torch._foreach_mul_(norm_params, 2)
                    torch._foreach_sub_(norm_params, mean_norm_params)
                    torch._foreach_div_(norm_params, ns)
                    torch._foreach_div_(norm_params, torch._foreach_mul(std_params, 2))

                    torch._foreach_sub_(std_params, device_kappas)
                    torch._foreach_mul_(std_params, device_kappa_updates)
                    torch._foreach_add_(device_lagmuls, std_params)
                    for lagmul in device_lagmuls:
                        lagmul.clip_(min=0.)
                    if reg_by_lr:
                        torch._foreach_addcmul_(device_params, norm_params, device_lagmuls, -1 * lr)
                    else:
                        torch._foreach_addcmul_(device_params, norm_params, device_lagmuls, -1)

                elif reg_function == 'l1':
                    abs_params = torch._foreach_abs(device_params)
                    abs_sum_params = []
                    for abs_param in abs_params:
                        abs_sum_params.append(abs_param.sum().unsqueeze(0))
                    torch._foreach_sub_(abs_sum_params, device_kappas)
                    torch._foreach_mul_(abs_sum_params, device_kappa_updates)
                    torch._foreach_add_(device_lagmuls, abs_sum_params)
                    for lagmul in device_lagmuls:
                        lagmul.clip_(min=0.)
                    if reg_by_lr:
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -1 * lr)
                    else:
                        torch._foreach_addcmul_(device_params, device_params, device_lagmuls, -1)

                elif reg_function == 'huber':
                    abs_params = torch._foreach_abs(device_params)
                    square_params = torch._foreach_pow(device_params, 2)
                    huber_loss_params, huber_loss_grads = [], []
                    for param_abs, square_param, device_param in zip(abs_params, square_params, device_params):
                        huber_loss_params.append(torch.where(param_abs < 1, 0.5 * square_param, param_abs - 0.5).sum())
                        huber_loss_grads.append(torch.where(param_abs < 1, device_param, device_param.sign()))
                    torch._foreach_sub_(huber_loss_params, device_kappas)
                    torch._foreach_mul_(huber_loss_params, device_kappa_updates)
                    torch._foreach_add_(device_lagmuls, huber_loss_params)
                    for lagmul in device_lagmuls:
                        lagmul.clip_(min=0.)
                    if reg_by_lr:
                        torch._foreach_addcmul_(device_params, huber_loss_grads, device_lagmuls, -1 * lr)
                    else:
                        torch._foreach_addcmul_(device_params, huber_loss_grads, device_lagmuls, -1)

                else:
                    raise ValueError(f"Unsupported regularization function: {reg_function}")

            if (kappa_init_method == 'inflection_point'
                    and any([device_kappa == HIGHKAPPA for device_kappa in device_kappas])):

                square_sum_params = torch._foreach_norm(device_params)

                if device_state_steps[0] >= reg_step_size:
                    torch._foreach_mul_(device_inflection_point_emas, reg_ema_decay)
                    torch._foreach_add_(device_inflection_point_emas, square_sum_params, alpha=(1 - reg_ema_decay))
                else:
                    torch._foreach_copy_(device_inflection_point_emas, square_sum_params)

                if device_state_steps[0] > reg_step_size * 1 and device_state_steps[0] % reg_step_size == 0:

                    current_gradients = torch._foreach_sub(device_inflection_point_emas, device_prev_regs)
                    current_gradients = torch._foreach_div(current_gradients, device_inflection_point_emas)
                    # Update previous values for next iteration
                    torch._foreach_copy_(device_prev_regs, device_inflection_point_emas)

                    # Peak detection for gradient
                    if device_state_steps[0] > reg_step_size * 3:
                        for i in range(len(device_params)):
                            if device_prev_reg_gradients[i] > current_gradients[i] > 0.01 \
                                    and device_kappas[i] == HIGHKAPPA:
                                single_initilize_kappa(device_kappas[i], device_params[i], reg_function)

                    if device_state_steps[0] > reg_step_size * 2:
                        torch._foreach_copy_(device_prev_reg_gradients, current_gradients)

            elif kappa_init_method == 'warm_start' and device_state_steps[0] == warm_start:

                if reg_function == 'l2':
                    square_sum_params = torch._foreach_pow(torch._foreach_norm(device_params), 2)
                    torch._foreach_add_(device_kappas, square_sum_params)

                elif reg_function == 'std':
                    new_kappas = [device_param.std() for device_param in device_params]
                    torch._foreach_add_(device_kappas, new_kappas)

                elif reg_function == 'l1':
                    abs_params = torch._foreach_abs(device_params)
                    new_kappas = [abs_param.sum() for abs_param in abs_params]
                    torch._foreach_add_(device_kappas, new_kappas)

                elif reg_function == 'huber':
                    abs_params = torch._foreach_abs(device_params)
                    square_params = torch._foreach_pow(device_params, 2)
                    new_kappas = [torch.where(param_abs < 1, 0.5 * square_param, param_abs - 0.5).sum() for
                                  param_abs, square_param in zip(abs_params, square_params)]
                    torch._foreach_add_(device_kappas, new_kappas)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamcpr)
def adamcpr(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        lagmuls: List[Tensor],
        kappas: List[Tensor],
        kappa_updates: List[Tensor],
        prev_regs: List[Tensor],
        prev_reg_gradients: List[Tensor],
        inflection_point_emas: List[Tensor],
        state_steps: List[Tensor],
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
        has_complex: bool = False,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[float, Tensor],
        regularize: bool,
        warm_start: int,
        reg_function: str,
        reg_by_lr: bool,
        kappa_init_method: str,
        reg_step_size: int,
        reg_ema_decay: float,
        eps: float,
        maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch.compiler.is_compiling() and not all(
            isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False

    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamcpr
    else:
        func = _single_tensor_adamcpr

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        lagmuls,
        kappas,
        kappa_updates,
        prev_regs,
        prev_reg_gradients,
        inflection_point_emas,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        regularize=regularize,
        warm_start=warm_start,
        reg_function=reg_function,
        reg_by_lr=reg_by_lr,
        kappa_init_method=kappa_init_method,
        reg_step_size=reg_step_size,
        reg_ema_decay=reg_ema_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
