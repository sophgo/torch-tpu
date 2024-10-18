import torch
import torch.nn as nn
import torch_tpu
from torch.optim.optimizer import _use_grad_for_differentiable

def _init_group(
    self,
    group,
    params_with_grad,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps
):
    for p in group['params']:
        if p.grad is not None:
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            grads.append(p.grad)

            state = self.state[p]
            # Lazy state initialization
            if len(state) == 0:
                state['step'] = (
                    torch.zeros((), dtype=torch.float, device=p.device)
                    if group['capturable'] or group['fused']
                    else torch.tensor(0., device=p.device)
                )
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)
                if group['amsgrad']:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])

            if group['amsgrad']:
                max_exp_avg_sqs.append(state['max_exp_avg_sq'])
            if group['differentiable'] and state['step'].requires_grad:
                raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

            # Foreach without capturable does not support a tensor lr
            if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

            state_steps.append(state['step'])

@_use_grad_for_differentiable
def tpu_adam_step(self, closure=None):
    """Performs a single optimization step.

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
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group['betas']
        
        _init_group(self,
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps)
        
        torch._fused_adam_(
            self=params_with_grad,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=state_steps,
            lr=group['lr'],
            beta1=beta1,
            beta2=beta2,
            weight_decay=group['weight_decay'],
            eps=group['eps'],
            amsgrad=group['amsgrad'],
            maximize=group['maximize'],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None)
        )
    return loss
            
def fuse_torch_adam():
    torch.optim.Adam.step = tpu_adam_step