import numpy as np
from typing import Any, Tuple, Optional

class Tensor:
    """
    Tensor class wrapping C++ core. Supports creation, elementwise ops, shape, device, and autograd hooks.
    """
    def __init__(self, data, requires_grad=False, device="cpu", dtype=None, _cpp_tensor=None):
        if _cpp_tensor is not None:
            self._data = _cpp_tensor
            self.shape = tuple(self._data.shape())
            self.device = self._data.device()
            self.dtype = self._data.dtype()
        else:
            self._data = np.array(data, dtype=dtype)
            self.shape = self._data.shape
            self.device = device
            self.dtype = self._data.dtype
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._ctx = None

    @staticmethod
    def ones(*shape, requires_grad=False, device="cpu", dtype=None):
        data = np.ones(shape, dtype=dtype)
        return Tensor(data, requires_grad, device, dtype)

    @staticmethod
    def zeros(*shape, requires_grad=False, device="cpu", dtype=None):
        data = np.zeros(shape, dtype=dtype)
        return Tensor(data, requires_grad, device, dtype)

    @staticmethod
    def randn(*shape, requires_grad=False, device="cpu", dtype=None):
        data = np.random.randn(*shape).astype(dtype or np.float32)
        return Tensor(data, requires_grad, device, dtype or np.float32)

    def numpy(self):
        return self._data

    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self._data + other._data)
        else:
            out = Tensor(self._data + other)
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self._data * other._data)
        else:
            out = Tensor(self._data * other)
        return out

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self._data @ other._data)
        else:
            out = Tensor(self._data @ other)
        return out

    def sum(self):
        return Tensor(self._data.sum())

    def backward(self, grad=None):
        from .autograd import backward
        backward(self, grad)

    def to(self, device):
        # For now, only CPU supported
        return self

    def __repr__(self):
        return f"Tensor({self._data}, requires_grad={self.requires_grad}, device={self.device})"
