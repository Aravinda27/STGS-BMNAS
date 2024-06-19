import torch

x = torch.randn(1024,257)
print(x)
print(x.shape)
y = x.view(8,128,257)
print(y)
print(y.shape)

