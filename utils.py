import os
os.chdir("..")
import torch
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_train_val_loaders(
    data_dir: str,
    x_file: str = "input_data.pt",
    y_file: str = "trues.pt",
    batch_size: int = 128,
    val_frac: float = 0.2,
    seed: int = 42,
)->Tuple[DataLoader, DataLoader]:
    """
    Loads tensors from .pt files (expects N x M x M inputs, N labels) and returns
    train and validation DataLoaders.
    """

    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)

    X = torch.load(x_path, map_location="cpu").float()
    y = torch.load(y_path, map_location="cpu").long()

    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape N x M x M, got {tuple(X.shape)}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to have shape (N,), got {tuple(y.shape)}")

    dataset = TensorDataset(X, y)

    N = len(dataset)
    n_val = int(val_frac * N)
    n_train = N - n_val
    train_data, val_data = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
