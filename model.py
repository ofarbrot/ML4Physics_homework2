import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class model(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()

        # --- DEFINING LAYERS ---
        None

        # --- WEIGHT LOADING ---
        if load_weights:
            try:
                path_to_py = os.path.dirname(__file__)

                weight_path = os.path.join(path_to_py, "model_weights.pth")

                state = torch.load(weight_path, map_location="cpu")
                self.load_state_dict(state)
                self.eval()

                print(f"Loaded pretrained weights from: {weight_path}")

            except FileNotFoundError:
                print("Warning: model_weights.pth not found, using random weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return None

    def pred(self, x): 

        return None