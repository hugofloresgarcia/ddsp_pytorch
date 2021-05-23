import ddsp
from ddsp.modules import Reverb, HarmonicSynth, FilteredNoise
from ddsp.models import GRUDecoder

from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class OperatorNet(nn.Module):

    def __init__(self, d_in: int, num_edges: int):
        super().__init__()
        self.num_edges = num_edges

        self.amp_proj = nn.Linear(d_in, num_edges)
        self.ratio_proj = nn.Linear(d_in, num_edges)

    def get_controls(self, x):
        """ 
        given an input embedding, projects x
        to:
            - an envelope vector of shape (block_size,)
            - an edge weight vector of shape (num_edges,)
            - a frequency ratio vector of of shape (num_edges,)
        """

        return {
            'amps': F.relu(self.amp_proj(x)+ 1),
            'ratios': F.relu(self.ratio_proj(x)+ 1) 
        }

class FMSynth(nn.Module):

    def __init__(self, d_in: int, block_size: int,
                 sample_rate: int, n_operators):
        super().__init__()
        self.block_size = block_size
        self.sample_rate = sample_rate

        self.operators = nn.ModuleList(
            [OperatorNet(d_in, n_operators) for _ in range(n_operators)]
            )
        self.carrier_weights = nn.Linear(d_in, n_operators)

    def forward(self, f0, x):
        """
        f0 (batch, seq, 1)
        x (batch, seq, d_in)
        """
        ctrls = [op.get_controls(x) for op in self.operators]
        amps = torch.stack(
            [ddsp.upsample(ctrl['amps'], self.block_size) for ctrl in ctrls])

        #maybe I want to predict discrete ratios in an exponential scale and add a learnable amount of small noise?
        ratios = torch.stack(
            [ddsp.upsample(ctrl['ratios'], self.block_size) for ctrl in ctrls])

        f0 = ddsp.upsample(f0, self.block_size).unsqueeze(0)

        omegas = torch.cumsum(2 * math.pi * f0 * ratios / self.sample_rate, 2)
        omega = torch.sum(
            torch.stack([a * torch.sin(w) for a, w in zip(amps, omegas)]),
            dim=-1, keepdim=True)

        omegas = torch.stack([omegas[idx, ..., idx] for idx in range(len(omegas))]).unsqueeze(-1)
        carriers = torch.sin(omega + omegas)
        # carriers = torch.sin(omegas)

        carrier_weights = torch.softmax(self.carrier_weights(x),dim=-1)
        carrier_weights = ddsp.upsample(carrier_weights, self.block_size).permute(2, 0, 1).unsqueeze(-1)
        
        sig = sum([cw* carrier for cw, carrier in zip(carriers, carrier_weights)])

        print('ratios', ratios.mean(dim=[1, 2, 3]))
        print('amps', amps.mean(dim=[1, 2, 3]))

        return {
            'signal': sig,
            'amps': amps,
            'ratios': ratios,
        }
