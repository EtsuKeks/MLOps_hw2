import torch
from torch.utils.data import Dataset, DataLoader
from swish import Swish
import pytorch_lightning as pl
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = Swish(x.tolist())  # Применяем биндинг
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        data = np.load(f"{self.data_path}/data.npy")
        labels = np.load(f"{self.data_path}/labels.npy")
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = torch.utils.data.random_split(
            CustomDataset(data, labels),
            [train_size, val_size]
        )
        self.train_dataset = train_data
        self.val_dataset = val_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
