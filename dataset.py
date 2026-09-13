from torch.utils.data import Dataset
import torch

torch.manual_seed(67)
x_dim = 1

class RegressionDataset(Dataset):
    def __init__(self, n, mean, std):
        self.n = n
        self.X = torch.rand((n, x_dim))

        self.y = self.X @ torch.rand((x_dim, 1)) + (
            torch.randn((n, 1)) * std + mean
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
