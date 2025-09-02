import numpy as np
from ..tensor import Tensor

class Module:
    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor.randn(in_features, out_features, requires_grad=True)
        self.bias = Tensor.randn(out_features, requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        return x @ self.weight + self.bias

class ReLU(Module):
    def forward(self, x):
        return Tensor(np.maximum(x._data, 0), requires_grad=x.requires_grad)

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
        self.bias = Tensor.randn(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        from .functional import conv2d
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)
