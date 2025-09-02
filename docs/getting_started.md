# Getting Started with MiniTorch

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/yourname/minitorch.git
cd minitorch
pip install -e .
```

## Build C++ Backend

```bash
mkdir build && cd build
cmake ..
make
```

## Example Usage

```python
import minitorch as mt
x = mt.Tensor.randn(64, 100, requires_grad=True)
w = mt.Tensor.randn(100, 10, requires_grad=True)
y = x @ w
loss = y.sum()
loss.backward()
print(w.grad.shape)
```

## Training a Model

```python
model = mt.nn.Sequential(
    mt.nn.Linear(784, 128),
    mt.nn.ReLU(),
    mt.nn.Linear(128, 10)
)
opt = mt.optim.Adam(model.parameters(), lr=1e-3)
for images, labels in loader:
    logits = model(images)
    loss = mt.nn.functional.cross_entropy(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
```
