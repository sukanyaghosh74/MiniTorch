import numpy as np
from collections import deque

class GraphNode:
    def __init__(self, op, inputs, grad_fn=None):
        self.op = op
        self.inputs = inputs
        self.grad_fn = grad_fn
        self.grad = None
        self.visited = False


def backward(tensor, grad=None):
    if grad is None:
        grad = np.ones_like(tensor._data)
    tensor.grad = grad
    queue = deque()
    queue.append((tensor, grad))
    visited = set()
    while queue:
        t, g = queue.popleft()
        if hasattr(t, '_grad_fn') and t._grad_fn is not None:
            grads = t._grad_fn(g)
            for inp, grad_inp in zip(t._ctx, grads):
                if inp.requires_grad:
                    if inp.grad is None:
                        inp.grad = grad_inp
                    else:
                        inp.grad += grad_inp
                    if inp not in visited:
                        queue.append((inp, grad_inp))
                        visited.add(inp)
