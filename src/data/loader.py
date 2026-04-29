import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)
        y = torch.randint(0, 4, (224, 224))
        return x, y
