import numpy as np
import threading
import queue

class Dataset:
    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = np.arange(len(dataset))
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.ptr = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.ptr:self.ptr+self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.ptr += self.batch_size
        return tuple(np.stack(x) for x in zip(*batch))
