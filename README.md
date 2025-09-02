# MiniTorch

MiniTorch is a **lightweight deep learning framework** inspired by PyTorch. It is designed to help users understand the inner workings of modern deep learning frameworks by providing a simplified yet powerful foundation for tensors, automatic differentiation, neural network layers, optimizers, and backends.

Unlike large-scale frameworks, MiniTorch emphasizes **transparency, modularity, and extensibility**. It offers both a **pure Python implementation** (for learning) and an **optimized C++ backend** (for performance) compiled via `pybind11`, with future support for GPU acceleration.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)

   * [Tensors](#tensors)
   * [Automatic Differentiation](#automatic-differentiation)
   * [Neural Network Layers](#neural-network-layers)
   * [Optimizers](#optimizers)
   * [Datasets and DataLoader](#datasets-and-dataloader)
5. [Examples](#examples)
6. [Architecture Overview](#architecture-overview)
7. [Performance](#performance)
8. [Development & Contribution](#development--contribution)
9. [Roadmap](#roadmap)
10. [License](#license)

---

## Features

* **Tensor API** – NumPy-like tensor operations with support for broadcasting, slicing, and matrix multiplication.
* **Autograd Engine** – Reverse-mode automatic differentiation for gradient-based optimization.
* **Layers & Models** – Build neural networks using layers like `Linear`, `Conv2d`, `ReLU`, `Sequential`.
* **Optimizers** – Implementations of `SGD`, `Adam`, and more.
* **Datasets & DataLoader** – Simple abstractions for dataset handling and mini-batching.
* **Python & C++ Backends** – Choose between pure Python (debuggable, transparent) and C++ (fast, optimized).
* **PyTorch-like API** – Easy transition for PyTorch users.
* **Educational Design** – Well-structured for learning the internals of deep learning systems.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sukanyaghosh74/MiniTorch.git
cd MiniTorch
```

### Python-only installation:

```bash
pip install -e .
```

### With C++ backend:

```bash
mkdir build && cd build
cmake ..
make
```

You can verify the installation by running:

```bash
python -c "import minitorch; print(minitorch.__version__)"
```

---

## Quick Start

### Tensor Operations

```python
import minitorch as mt

# Create random tensors
x = mt.Tensor.randn(3, 3, requires_grad=True)
y = mt.Tensor.ones(3, 3)

# Perform operations
z = (x * y).sum()

# Backward pass\ nz.backward()

print("Tensor value:", z.item())
print("Gradient shape:", x.grad.shape)
```

### Simple Neural Network

```python
import minitorch as mt

class SimpleNN(mt.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = mt.nn.Linear(10, 50)
        self.relu = mt.nn.ReLU()
        self.layer2 = mt.nn.Linear(50, 1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# Initialize model, loss, optimizer
model = SimpleNN()
criterion = mt.nn.MSELoss()
optimizer = mt.optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(5):
    inputs = mt.Tensor.randn(64, 10)
    targets = mt.Tensor.randn(64, 1)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---

## Core Concepts

### Tensors

* Multi-dimensional arrays with support for elementwise and matrix operations.
* Can track gradients when `requires_grad=True`.
* Similar to PyTorch’s `torch.Tensor`.

```python
a = mt.Tensor([1, 2, 3], requires_grad=True)
b = mt.Tensor([4, 5, 6])
c = (a * b).sum()
c.backward()
print(a.grad)  # → gradients for a
```

### Automatic Differentiation

* Uses **reverse-mode autodiff**.
* Builds a dynamic computation graph during forward pass.
* Calls `.backward()` to compute gradients.

### Neural Network Layers

Available layers include:

* `Linear`
* `Conv2d`
* `ReLU`
* `Sigmoid`
* `Sequential`

### Optimizers

Currently supported:

* `SGD` (with momentum)
* `Adam`

```python
optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
```

### Datasets and DataLoader

```python
from minitorch.utils import DataLoader, Dataset

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.data = mt.Tensor.randn(length, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(RandomDataset(10, 100), batch_size=16)
for batch in loader:
    print(batch.shape)
```

---

## Examples

### Training a Logistic Regression Model

```python
import minitorch as mt

class LogisticRegression(mt.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = mt.nn.Linear(in_features, out_features)

    def forward(self, x):
        return mt.nn.Sigmoid()(self.linear(x))

model = LogisticRegression(2, 1)
criterion = mt.nn.BCELoss()
optimizer = mt.optim.SGD(model.parameters(), lr=0.01)

# Dummy dataset
X = mt.Tensor.randn(100, 2)
y = mt.Tensor.randint(0, 2, (100, 1))

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---

## Architecture Overview

```
MiniTorch
├── minitorch/
│   ├── tensor.py        # Tensor operations & autograd
│   ├── nn/              # Neural network layers
│   ├── optim/           # Optimizers
│   ├── utils/           # DataLoader, helpers
│   └── backend/         # Python and C++ backends
├── tests/               # Unit tests
├── examples/            # Example models & scripts
└── build/               # Compiled backend (optional)
```

* **Tensor Engine**: Core class supporting computation graph construction.
* **Autograd Engine**: Traverses graph backwards to compute gradients.
* **nn Module**: Defines neural layers and activation functions.
* **Optim Module**: Houses optimizers.
* **Backends**: Switchable execution engine (Python/C++).

---

## Performance

* Pure Python backend is \~10x slower than PyTorch.
* C++ backend reduces gap to \~1.2x–1.3x slower on MNIST and CIFAR-10 benchmarks.
* Suitable for educational and prototyping purposes.

---

## Development & Contribution

### Repository Structure

* `minitorch/` → main source code
* `tests/` → unit tests using `pytest`
* `examples/` → runnable demos and notebooks

### Running Tests

```bash
pytest tests/
```

### Style Guide

* Follow PEP8 for Python code.
* Use `black` for formatting.
* Use docstrings for public methods.

### Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR with detailed description

---

## Roadmap

* [ ] GPU backend via CUDA
* [ ] More layers (BatchNorm, Dropout, RNNs)
* [ ] Expanded optimizer library
* [ ] ONNX export support
* [ ] Visualization tools for computation graphs

---

## License

Apache License 2.0. See \[LIC]
