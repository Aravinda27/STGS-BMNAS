import torch
import torch.nn as nn
from models.central import ntu

x = torch.randn(3,600,257)
model = ntu.Spectrum()
f1,f2, f3, f4,f5,f6 = model(x)
print(f1.shape)
print(f2.shape)
print(f3.shape)
print(f4.shape)
print(f5.shape)
print(f6.shape)