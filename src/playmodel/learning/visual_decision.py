"""Game-independent trainable visual context for variable decision candidates.

No keys, enemy classes, rewards, or game rules live here. A short causal window
is replayed from zero memory for every choice, including training replay.
"""
from __future__ import annotations

import math
import torch
from torch import nn

SCHEMA = 'playmodel.visual-goal.v1'


class VisualDecisionContext(nn.Module):
    def __init__(self, candidate_dim: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 6 * 6, hidden_size), nn.ReLU())
        self.temporal = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.project = nn.Linear(hidden_size, candidate_dim, bias=False)
        # Exact legacy policy on migration. First update learns the projection;
        # subsequent updates can train the visual encoder and temporal memory.
        nn.init.zeros_(self.project.weight)
        self.candidate_norm = nn.LayerNorm(candidate_dim, elementwise_affine=False)
        self.object_encoder = nn.Sequential(nn.Conv2d(3,16,5,2,2), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, hidden_size), nn.ReLU())
        self.object_project = nn.Linear(hidden_size, candidate_dim, bias=False)
        nn.init.zeros_(self.object_project.weight)

    def forward(self, frames, candidates, object_frames=None):
        if frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[2:] != (3, 96, 96):
            raise ValueError('visual context requires RGB uint8 [batch,time,3,96,96]')
        batch, count = frames.shape[:2]
        if not 1 <= count <= 4 or candidates.ndim != 3 or candidates.shape[0] != batch:
            raise ValueError('invalid visual context/candidate dimensions')
        features = self.encoder(frames.reshape(-1, 3, 96, 96).float() / 255)
        # No persistent hidden state: replay sees exactly the same window.
        memory, _ = self.temporal(features.reshape(batch, count, -1))
        query = self.project(memory[:, -1])
        if object_frames is not None:
            if (object_frames.dtype != torch.uint8 or object_frames.ndim != 5
                    or object_frames.shape[0] != batch or object_frames.shape[2:] != (3,32,32)
                    or not 1 <= object_frames.shape[1] <= 4):
                raise ValueError('object context requires RGB uint8 [batch,objects,3,32,32]')
            objects = self.object_encoder(object_frames.reshape(-1,3,32,32).float()/255)
            query = query + self.object_project(objects.reshape(batch,-1,objects.shape[-1]).mean(1))
        return torch.einsum('bd,bkd->bk', query, self.candidate_norm(candidates)) / math.sqrt(query.shape[-1])
