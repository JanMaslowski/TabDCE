import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant', value=0.0)
        return emb

class FiLM(nn.Module):
    def __init__(self, hdim: int, cdim: int):
        super().__init__()
        self.to_params = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, hdim * 2)
        )

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.to_params(c)
        gamma, beta = params.chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, dim: int, cdim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(dim)
        self.film = FiLM(hidden, cdim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.film(h, c) 
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return self.act(h + res)

class TabularEpsModel(nn.Module):
    def __init__(self, latent_dim: int, y_classes: int, hidden: int = 256, nblocks: int = 4, tdim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.time_emb = SinusoidalTimeEmbedding(tdim)
        self.y_emb = nn.Embedding(y_classes + 1, 64)
        
        self.inp_proj = nn.Linear(latent_dim * 2 + 64, hidden)
        self.cond_dim = tdim + 64

        self.blocks = nn.ModuleList([
            ResBlock(hidden, self.cond_dim, hidden, dropout) for _ in range(nblocks)
        ])
        
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_orig: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)
        yemb = self.y_emb(y_target)
        cond = torch.cat([temb, yemb], dim=-1)

        h = torch.cat([x_t, z_orig, yemb], dim=-1)
        h = self.inp_proj(h)
        
        for blk in self.blocks:
            h = blk(h, cond)
            
        out = self.out_proj(h)
        return out