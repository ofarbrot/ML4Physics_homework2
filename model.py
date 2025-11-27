import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class model(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()

        # --- DEFINING LAYERS ---
        # Anta input-shape: (N, T). Vi legger til en kanal-dimensjon i forward: (N, 1, T)
        self.conv_net = nn.Sequential(
            # First conv: see local correlations over a window of 5 time steps
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),

            # Second conv: build on first-level features
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),

            # Third conv: slightly wider kernel to catch longer correlations
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
        )
        """
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )"""

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)

        # Komprimerer over tid til lengde 1, slik at vi ender med (N, 64, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected del for regresjon til én skalar
        """        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)"""
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)   # raw scalar, will be squashed to [0,2] in forward()
        )

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

    def forward1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forventer x med shape (N, T) eller (T,) for én enkelt traj.
        Returnerer tensor med shape (N,) med predikerte alfabetraktorer.
        """

        # Sørg for batch-dimensjon
        if x.ndim == 1:
            # Én enkelt tidsserie: (T,) -> (1, T)
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"Expected input of shape (N, T) or (T,), got {x.shape}")

        # Legg til kanal-dimensjon: (N, T) -> (N, 1, T)
        x = x.unsqueeze(1)

        # CNN-del
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global gjennomsnittspooling over tid: (N, 64, L) -> (N, 64, 1)
        x = self.global_pool(x)

        # Flatten: (N, 64, 1) -> (N, 64)
        x = x.squeeze(-1)

        # FC-del
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (N, 1)

        # Returner (N,)
        return x.squeeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Add channel dimension: (B, T) → (B, 1, T)
        x = x.unsqueeze(1)

        # Extract temporal features with 1D convs
        features = self.conv_net(x)            # (B, 64, T)

        # Global average pool over time: (B, 64, T) → (B, 64, 1) → (B, 64)
        pooled = self.global_pool(features).squeeze(-1)

        # Map to a scalar
        alpha_raw = self.regressor(pooled).squeeze(-1)  # (B,)

        # Constrain to [0, 2] by using a sigmoid and scaling
        alpha = 2.0 * torch.sigmoid(alpha_raw)

        return alpha

    def pred(self, x: torch.Tensor) -> torch.Tensor:
        """
        Required by platform: tar inn data med samme format som trening (N, T)
        og returnerer 1D-tensor med predikerte alfabetraktorer (N,).
        """
        self.eval()
        x = x.to(torch.float32)

        with torch.no_grad():
            preds = self.forward(x)

        return preds