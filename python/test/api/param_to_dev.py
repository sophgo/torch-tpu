import torch
import copy
import torch_tpu
TPU = "tpu"
#TPU = "cuda" 

class RandomModule(torch.nn.Module):
    def __init__(self):
        super(RandomModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8).half())
        self.bias = torch.nn.Parameter(torch.randn(8).half())

    def forward(self, input):
        return input * self.weight + self.bias

model = RandomModule()

print("Get a random model")
print("Add something to model parameters")
for name, param in model.named_parameters():
    if not hasattr(param, 'something'):
        setattr(param, 'something', 1)
        print(" added something 1 to param", name)

print("Check if something is added in CPU model")
for name, param in model.named_parameters():
    if hasattr(param, 'something'):
        print(f" param {name}  id {id(param)} has something", param.something)
    else:
        print(f" param {name}  id {id(param)} does not have something")

model_ = copy.copy(model)
print("Check if something is added in CPU model")
for name, param in model_.named_parameters():
    if hasattr(param, 'something'):
        print(f" param {name}  id {id(param)} has something", param.something)
    else:
        print(f" param {name}  id {id(param)} does not have something")


'''
Output:
Check if something is added in CPU model
 param weight has something 1
 param bias has something 1
'''
print("Convert to TPU")
import pdb;pdb.set_trace()
model_tpu = model.to(TPU)

print("Check if something is added in TPU model")
for name, param in model_tpu.named_parameters():
    if hasattr(param, 'something'):
        print(f" param {name} id {id(param)} has something", param.something)
    else:
        print(f" param {name} id {id(param)} does not have something")
'''
Output:
Check if something is added in TPU model
 param weight does not have something
 param bias does not have something
'''


# a = torch.rand([10])
# a = torch.nn.Parameter(a)
# setattr(a, 'something', 1)
# print(f"id a : {id(a)}, a.device = {a.device}")
# b = a.to(TPU)
# print("=========after======")
# print(f"id a : {id(a)}, a.device = {a.device}, has_something = {hasattr(a, 'something')}")
# print(f"id b : {id(b)}, b.device = {b.device}, has_something = {hasattr(b, 'something')}")
