import torch
from torch.utils.data import Dataset, DataLoader

class AutocompleteDataset(Dataset):
    def __init__(self, X, y, preprocessor):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)