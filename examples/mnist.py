import minitorch as mt
from minitorch.data import DataLoader, Dataset
import numpy as np

class MNIST(Dataset):
    def __init__(self, train=True):
        # Dummy data for demo
        self.images = np.random.randn(100, 1, 28, 28).astype(np.float32)
        self.labels = np.random.randint(0, 10, 100)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    def __len__(self):
        return len(self.images)

dataset = MNIST()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = mt.nn.Sequential(
    mt.nn.Conv2d(1, 8, 3),
    mt.nn.ReLU(),
    mt.nn.Linear(8*26*26, 10)
)
opt = mt.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    for images, labels in loader:
        images = images.reshape(images.shape[0], 1, 28, 28)
        x = mt.Tensor(images, requires_grad=True)
        logits = model(x)
        logits = mt.Tensor(logits._data.reshape(images.shape[0], 10), requires_grad=True)
        loss = mt.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} loss: {loss._data}")
