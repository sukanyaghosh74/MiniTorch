import minitorch as mt
import numpy as np

def test_backward():
    x = mt.Tensor.randn(3, 3, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad.shape == x._data.shape

def test_grad_accumulation():
    x = mt.Tensor.randn(2, 2, requires_grad=True)
    y1 = (x * 2).sum()
    y2 = (x * 3).sum()
    y1.backward()
    grad1 = x.grad.copy()
    x.grad = None
    y2.backward()
    grad2 = x.grad.copy()
    assert not np.allclose(grad1, grad2)
