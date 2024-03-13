import inspect

from .optim_cpr import CPR
from .group_parameter import cpr_group_named_parameters


def apply_CPR(model, optimizer_cls, kappa_init_param, kappa_init_method='warm_start', reg_function='l2',
              kappa_adapt=False, kappa_update=1.0,
              normalization_regularization=False, bias_regularization=False, embedding_regularization=False,
              **optimizer_args):
    optimizer_args['weight_decay'] = 0
    avoid_keywords = []

    param_groups = cpr_group_named_parameters(model=model, optim_hps=optimizer_args, avoid_keywords=avoid_keywords,
                                              embedding_regularization=embedding_regularization,
                                              bias_regularization=bias_regularization,
                                              normalization_regularization=normalization_regularization)

    optimizer_keys = inspect.getfullargspec(optimizer_cls).args
    for k, v in optimizer_args.items():
        if k not in optimizer_keys:
            raise UserWarning(f"apply_CPR: Unknown optimizer argument {k}")
    optimizer = optimizer_cls(param_groups, **optimizer_args)

    optimizer = CPR(optimizer=optimizer, kappa_init_param=kappa_init_param, kappa_init_method=kappa_init_method,
                    reg_function=reg_function, kappa_adapt=kappa_adapt, kappa_update=kappa_update)

    return optimizer
