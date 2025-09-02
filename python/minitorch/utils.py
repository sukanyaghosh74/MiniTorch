import numpy as np

def set_seed(seed):
    np.random.seed(seed)

# Gradient check for scalar-valued function

def grad_check(f, x, eps=1e-5, tol=1e-4):
    x = x.copy()
    fx = f(x)
    grad_approx = np.zeros_like(x)
    for i in range(x.size):
        x[i] += eps
        fx1 = f(x)
        x[i] -= 2 * eps
        fx2 = f(x)
        x[i] += eps
        grad_approx[i] = (fx1 - fx2) / (2 * eps)
    return grad_approx
