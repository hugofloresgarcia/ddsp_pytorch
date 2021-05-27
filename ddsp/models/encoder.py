import ddsp

import librosa
import torch
import torch.nn as nn

class MFCCEncoder(nn.Module):

    def __init__(self, sample_rate: int, block_size: int,
                 hidden_size: int, n_mfccs: int, z_dim: int = None):
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.norm = nn.LayerNorm(n_mfccs)
        self.gru = ddsp.gru(n_mfccs, hidden_size)
        self.proj = nn.Linear(hidden_size, z_dim)


    def forward(self, mfccs: torch.Tensor):
        x = self.norm(mfccs)
        x, _ = self.gru(x)
        z = self.proj(x)
        breakpoint()
        return z
