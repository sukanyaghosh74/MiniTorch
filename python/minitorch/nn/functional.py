import numpy as np
from ..tensor import Tensor

def relu(x):
    return Tensor(np.maximum(x._data, 0), requires_grad=x.requires_grad)

def conv2d(x, weight, bias, stride=1, padding=0):
    # Naive implementation for demonstration
    N, C, H, W = x._data.shape
    F, _, HH, WW = weight._data.shape
    out_h = (H + 2 * padding - HH) // stride + 1
    out_w = (W + 2 * padding - WW) // stride + 1
    out = np.zeros((N, F, out_h, out_w), dtype=x._data.dtype)
    x_padded = np.pad(x._data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    for n in range(N):
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    window = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    out[n, f, i, j] = np.sum(window * weight._data[f]) + bias._data[f]
    return Tensor(out, requires_grad=x.requires_grad or weight.requires_grad or bias.requires_grad)

def cross_entropy(logits, targets):
    # logits: (N, C), targets: (N,)
    exps = np.exp(logits._data - np.max(logits._data, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    N = logits._data.shape[0]
    loss = -np.log(probs[np.arange(N), targets])
    return Tensor(loss.mean(), requires_grad=logits.requires_grad)
