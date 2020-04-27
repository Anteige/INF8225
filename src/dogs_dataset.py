import torch
from torch.utils.data import Dataset

class DogsDataset(Dataset):
    """ Dogs dataset."""
    # Initialize your data, download, etc.
    def __init__(self, X, Y):
        self.len = Y.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).float()
        print(self.x_data.shape)
        print(self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len