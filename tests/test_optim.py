import minitorch as mt
import numpy as np

def test_sgd_step():
    x = mt.Tensor.randn(2, 2, requires_grad=True)
    x.grad = np.ones((2, 2))
    opt = mt.optim.SGD([x], lr=0.1)
    old = x._data.copy()
    opt.step()
    assert not np.allclose(x._data, old)

def test_adam_step():
    x = mt.Tensor.randn(2, 2, requires_grad=True)
    x.grad = np.ones((2, 2))
    opt = mt.optim.Adam([x], lr=0.1)
    old = x._data.copy()
    opt.step()
    assert not np.allclose(x._data, old)
