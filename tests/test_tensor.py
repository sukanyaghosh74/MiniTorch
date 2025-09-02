import pytest
import minitorch as mt

def test_tensor_creation():
    t = mt.Tensor.ones(2, 3)
    assert t.shape == (2, 3)
    assert t._data.shape == (2, 3)

def test_tensor_ops():
    a = mt.Tensor.ones(2, 2)
    b = mt.Tensor.ones(2, 2)
    c = a + b
    assert (c._data == 2).all()
    d = a * b
    assert (d._data == 1).all()
    e = a @ b
    assert e._data.shape == (2, 2)

def test_tensor_sum():
    t = mt.Tensor.ones(4)
    s = t.sum()
    assert s._data == 4
