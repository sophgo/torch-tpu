import deepspeed

def optim_wrapper(func):
    def wrapper(self, client_optimizer, model_parameters):
        if str(self.device).startswith('tpu'):
            if hasattr(client_optimizer, 'param_groups'):
                used = [False for param in model_parameters]
                for group in client_optimizer.param_groups:
                    for i in range(len(group['params'])):
                        changed = False
                        for idx, param in enumerate(model_parameters):
                            if not used[idx] and group['params'][i].shape == param.shape and (group['params'][i] == param.cpu()).all():
                                group['params'][i] = param
                                used[idx] = True
                                changed = True
                                break
                        assert changed, "Optimizer param not found in model"
        return func(self, client_optimizer, model_parameters)
    return wrapper
deepspeed.runtime.engine.DeepSpeedEngine._configure_optimizer = optim_wrapper(deepspeed.runtime.engine.DeepSpeedEngine._configure_optimizer)
