import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class ConditionalMLPDiffusion(nn.Module):
    def __init__(self, d_in, y_classes, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in * 2 + 64, dim_t)
        self.y_emb = nn.Embedding(y_classes + 1, 64)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t* 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x_t, noise_labels, z_orig, y_target):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) 
        emb = self.time_embed(emb)
        y_e = self.y_emb(y_target)
        x = torch.cat([x_t, z_orig, y_e], dim=-1)
        x = self.proj(x) + emb
        delta = self.mlp(x)
        return z_orig + delta


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,               
        sigma_max = float('inf'),    
        sigma_data = 0.5,            
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, z_orig, y_target):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.flatten(), z_orig, y_target)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

class LatentTabularDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module, 
        vae_model: nn.Module, 
        latent_dim: int,
        T: int = 50,           
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.vae = vae_model
        self.latent_dim = latent_dim
        self.device = device
        self.T_steps = T 
        
        y_classes = max(2, len(self.vae.cat_cardinalities)) 

        core_net = ConditionalMLPDiffusion(d_in=latent_dim, y_classes=y_classes)
        self.precond = Precond(denoise_fn=core_net, hid_dim=latent_dim, sigma_data=0.5)
        
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.register_buffer('scale_initialized', torch.tensor(0))

    def forward(self, x_neigh: torch.Tensor, x_orig: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        device = x_neigh.device 
        B = x_neigh.shape[0]
        
        with torch.no_grad():
            z_neigh = self.vae.encode(x_neigh) 
            z_orig  = self.vae.encode(x_orig) 
        if self.scale_initialized.item() == 0:
            current_std = z_neigh.std().clamp(min=1e-6)
            self.scale_factor.data = self.sigma_data / current_std
            self.scale_initialized.data = torch.tensor(1)
            print(f"\n[INFO] VAE Latent ustabilizowany! Ustawiono mnożnik skali: {self.scale_factor.item():.4f}\n")

        z_neigh = z_neigh * self.scale_factor
        z_orig = z_orig * self.scale_factor

        rnd_normal = torch.randn(B, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        noise = torch.randn_like(z_neigh) * sigma.unsqueeze(1)
        z_t = z_neigh + noise
        
        drop_z_prob = 0.2
        drop_z_mask = torch.rand(B, device=device) < drop_z_prob
        z_orig_cond = z_orig.clone()
        z_orig_cond[drop_z_mask] = 0.0  
        
        drop_y_prob = 0.15
        null_token_id = self.precond.denoise_fn_F.y_emb.num_embeddings - 1 
        drop_y_mask = torch.rand(B, device=device) < drop_y_prob
        
        y_train_cond = y_target.clone()
        y_train_cond[drop_y_mask] = null_token_id

        D_x = self.precond(z_t, sigma, z_orig_cond, y_train_cond)
        
        loss = weight.unsqueeze(1) * ((D_x - z_neigh) ** 2)
        return loss.mean()
    def get_sigmas_karras(self, n, sigma_min=0.002, sigma_max=80.0, rho=7.0, device=None):
        if device is None:
            device = self.device
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    @torch.no_grad()
    def sample(
        self,
        x_orig: torch.Tensor,
        y_target: torch.Tensor,
        temperature: float = 1.0,  
        guidance_scale: float = 5.0
    ) -> torch.Tensor:
        
        device = next(self.parameters()).device
        x_orig = x_orig.to(device)
        y_target = y_target.to(device)
        B = x_orig.shape[0]
        z_orig = self.vae.encode(x_orig) * self.scale_factor
        
        sigmas = self.get_sigmas_karras(self.T_steps, device=device)
        x = torch.randn(B, self.latent_dim, device=device) * sigmas[0] * temperature
        
        null_token_id = self.precond.denoise_fn_F.y_emb.num_embeddings - 1
        y_null = torch.full_like(y_target, null_token_id, device=device)
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_tensor = torch.full((B,), sigma.item(), device=device)
            
            if guidance_scale > 1.0:
                D_uncond = self.precond(x, sigma_tensor, z_orig, y_null)
                D_cond = self.precond(x, sigma_tensor, z_orig, y_target)
                D_x = D_uncond + guidance_scale * (D_cond - D_uncond)
            else:
                D_x = self.precond(x, sigma_tensor, z_orig, y_target)

            d = (x - D_x) / sigma
            x = x + d * (sigma_next - sigma)
            
        x = x / self.scale_factor
        x_cf = self.vae.decode(x)
        return x_cf