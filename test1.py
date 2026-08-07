import torch

key = torch.Generator(46)
parameters = torch.randn((27,27), dtype = torch.int64)

print(parameters)