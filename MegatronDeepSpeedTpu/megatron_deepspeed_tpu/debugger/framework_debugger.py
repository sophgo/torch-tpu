import deepspeed
import megatron
from megatron import get_args
from megatron.training import train_step

from .module_debugger import save_model_params, save_tensors, combine_npz, print_log

def backward_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        save_tensors(args[0].bit16_groups, f'bit16_groups_grads', dir="debug", save_grad_instead=True)
        return ret
    return wrapper
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.backward = backward_wrapper(deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.backward)

def train_step_wrapper(func):
    def wrapper(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
        iter = get_args().curr_iteration
        if iter == 0:
            save_model_params(model[0], 0)
        ret = func(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config)
        save_model_params(model[0], iter + 1)
        combine_npz(iter + 1)
        print_log(f"Loss: {ret[0]['lm loss'].item()}")
        return ret
    return wrapper
megatron.training.train_step = train_step_wrapper(train_step)

###############################################################################################

###############################################################################################
# dev
def tdb_after_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        import pdb;pdb.set_trace()
        return ret
    return wrapper

def tdb_before_wrapper(func):
    def wrapper(*args, **kwargs):
        import pdb;pdb.set_trace()
        ret = func(*args, **kwargs)
        return ret
    return wrapper