class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = [0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] + p.grad
                p._data -= self.lr * self.v[i]

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [0 for _ in self.params]
        self.v = [0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad ** 2)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                p._data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
