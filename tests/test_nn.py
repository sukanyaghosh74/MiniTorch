import minitorch as mt
import numpy as np

def test_linear_forward():
    m = mt.nn.Linear(4, 3)
    x = mt.Tensor.randn(2, 4)
    y = m(x)
    assert y._data.shape == (2, 3)

def test_relu_forward():
    relu = mt.nn.ReLU()
    x = mt.Tensor(np.array([[-1, 2], [3, -4]]))
    y = relu(x)
    assert (y._data >= 0).all()

def test_sequential():
    model = mt.nn.Sequential(mt.nn.Linear(4, 4), mt.nn.ReLU(), mt.nn.Linear(4, 2))
    x = mt.Tensor.randn(2, 4)
    y = model(x)
    assert y._data.shape == (2, 2)
