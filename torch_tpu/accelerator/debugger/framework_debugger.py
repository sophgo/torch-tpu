from functools import wraps
import torch
import time
from nnmoduletools.module_debugger import register_hook_for_Function, save_model_params, save_model_grads, save_tensors, combine_npz, print_log
#############################
# deepspeed

try:
    import deepspeed
    def backward_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            save_tensors(args[0].bit16_groups, f'bit16_groups_grads', dir="debug", save_grad_instead=True)
            return ret
        return wrapper
    deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.backward = backward_wrapper(deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.backward)
except ImportError:
    pass

#############################
# megatron

def train_step_wrapper(func):
    @wraps(func)
    def wrapper(forward_step_func, data_iterator,
               model, *args, **kwargs):
        iter = get_args().curr_iteration
        if iter == 0:
            save_model_params(model[0].module, 0)
        timestamp = time.time()
        ret = func(forward_step_func, data_iterator,
               model, *args, **kwargs)
        if 'lm loss' in ret[0]:
            print_log(f"Loss: {ret[0]['lm loss'].item()}")
        elapsed_time = time.time() - timestamp
        print_log(f"Time elapsed for iter {iter}: {elapsed_time}s")
        save_model_params(model[0].module, iter + 1)
        combine_npz(iter + 1)
        return ret
    return wrapper

try:
    import megatron
    try:
        from megatron import get_args # megatron-deepspeed
        from megatron.training import train_step
        
        # save model params and grads
        megatron.training.train_step = train_step_wrapper(train_step)
        
        # will get error view+inplace when using hook
        def attention_mask_func(attention_score, attention_mask):
            args = get_args()
            if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
                attention_mask_ = attention_mask
                actual_seqlen = attention_scores.size()[2]
                if actual_seqlen != attention_mask_.size()[2]:
                    # attention_mask has size [1, 1, seqlen, seqlen]
                    attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
                attention_scores = attention_scores.masked_fill(attention_mask_, -10000.0)
            else:
                attention_scores = attention_scores.masked_fill(attention_mask, -10000.0)
            return attention_scores
        
        megatron.model.transformer.attention_mask_func = attention_mask_func
        
    except ImportError:
        from megatron.training import get_args # megatron-lm
        from megatron.training.training import train_step
        from megatron import core
        
        # save model params
        megatron.training.training.train_step = train_step_wrapper(train_step)
        # save model grads
        def get_fb_func_wrapper(func):
            @wraps(func)
            def wrapper():
                fw_func_orig = func()
                def fw_func(*args, **kwargs):
                    ret = fw_func_orig(*args, **kwargs)
                    iter = get_args().curr_iteration
                    save_model_grads(kwargs['model'][0].module, iter + 1, grad_attr="main_grad")
                    return ret
                return fw_func
            return wrapper
        megatron.training.training.get_forward_backward_func = get_fb_func_wrapper(megatron.core.pipeline_parallel.schedules.get_forward_backward_func)

        # save results of all reduce
        # megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication = register_hook_for_Function(megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication)
        
        # will get error view+inplace when using hook
        def attention_mask_func(attention_scores, attention_mask):
            attention_scores = attention_scores.masked_fill(attention_mask, -10000.0)
            return attention_scores
        
        megatron.core.transformer.utils.attention_mask_func = attention_mask_func
        megatron.core.transformer.dot_product_attention.attention_mask_func = attention_mask_func
        
        # will get error in pipeline parallel when using hook
        def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
            if (out is None) or (not deallocate_pipeline_outputs):
                return
            assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
            # assert out._base is None, "counter-productive to free a view of another tensor."
            out.data = torch.empty(
                (1,),
                device=out.device,
                dtype=out.dtype,
            )
            if out._base is not None:
                out._base.data = out.data
        megatron.core.pipeline_parallel.schedules.deallocate_output_tensor = deallocate_output_tensor
          
except ImportError:
    pass

###############################################################################################

###############################################################################################
# dev
def tdb_after_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        import pdb;pdb.set_trace()
        return ret
    return wrapper

def tdb_before_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import pdb;pdb.set_trace()
        ret = func(*args, **kwargs)
        return ret
    return wrapper
# megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_torch_softmax = tdb_before_wrapper(megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_torch_softmax)