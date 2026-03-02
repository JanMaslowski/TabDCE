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

class ResidualBlock(nn.Module):
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)
        
        # Prawdziwe FiLM wymaga DWÓCH wektorów sterujących dla każdego neuronu (Gamma i Beta)
        # Dlatego wyjście to dim * 2
        self.emb_proj = nn.Linear(emb_dim, dim * 2)

    def forward(self, x, emb):
        # Generujemy suwaki (Gamma) i przesunięcia (Beta) z warunku
        emb_out = self.emb_proj(emb)
        gamma, beta = emb_out.chunk(2, dim=-1)
        
        # MAGIA FiLM: Brutalne mnożenie neuronów przez warunek!
        # Jeśli warunek chce zgasić jakąś cechę, gamma będzie bliskie -1 (wtedy 1 + gamma = 0)
        h = x * (1 + gamma) + beta
        
        return x + self.linear2(self.act(self.linear1(h)))

class ConditionalMLPDiffusion(nn.Module):
    def __init__(self, d_in, y_classes, dim_t=128):
        super().__init__()
        
        # Mniejsza, zgrabniejsza sieć: 256 neuronów na warstwę
        self.dim_t = max(256, d_in * 2) 
        
        self.proj = nn.Linear(d_in * 2 + 64, self.dim_t)
        self.y_emb = nn.Embedding(y_classes + 1, 64)

        self.map_noise = PositionalEmbedding(num_channels=self.dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t)
        )
        
        # Projektor do FiLM
        self.cond_proj = nn.Sequential(
            nn.Linear(self.dim_t + 64, self.dim_t),
            nn.SiLU()
        )

        # Używamy naszych klocków z FiLM!
        self.blocks = nn.ModuleList([
            ResidualBlock(self.dim_t, emb_dim=self.dim_t),
            ResidualBlock(self.dim_t, emb_dim=self.dim_t),
            ResidualBlock(self.dim_t, emb_dim=self.dim_t)
        ])
        
        self.out_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim_t, d_in)
        )
    
    def forward(self, x_t, noise_labels, z_orig, y_target):
        # 1. Przygotowanie wejścia
        x = torch.cat([x_t, z_orig, self.y_emb(y_target)], dim=-1)
        h = self.proj(x)
        
        # 2. Przygotowanie WARUNKU (Czas + Klasa) dla FiLM
        t_emb = self.time_embed(self.map_noise(noise_labels))
        cond = self.cond_proj(torch.cat([t_emb, self.y_emb(y_target)], dim=-1))
        
        # 3. Przepuszczamy przez bloki rezydualne
        for block in self.blocks:
            h = block(h, cond)
            
        return self.out_layer(h)

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
        sigma = (rnd_normal * self.P_std + self.P_mean).exp().clamp(min=1e-3)
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2).clamp(max=100.0)
        
        noise = torch.randn_like(z_neigh) * sigma.unsqueeze(1)
        z_t = z_neigh + noise
        
        # CZYSTY TRENING WARUNKOWY - Bez dropoutu, model zawsze widzi z_orig i y_target!
        D_x = self.precond(z_t, sigma, z_orig, y_target)
        
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
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        
        device = next(self.parameters()).device
        x_orig = x_orig.to(device)
        y_target = y_target.to(device)
        B = x_orig.shape[0]
        
        # Kodujemy oryginał
        z_orig = self.vae.encode(x_orig) * self.scale_factor
        
        # MAGIA: Startujemy z szumu, na którym model faktycznie trenował!
        sigmas = self.get_sigmas_karras(self.T_steps, sigma_max=5.0, device=device)
        
        # Zaczynamy od szumu dopasowanego do treningu
        x = torch.randn(B, self.latent_dim, device=device) * sigmas[0] * temperature
        
        # Pętla odszumiająca
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_tensor = torch.full((B,), sigma.item(), device=device)
            
            # Tylko jedno przejście przez sieć w kierunku y_target
            D_x = self.precond(x, sigma_tensor, z_orig, y_target)

            d = (x - D_x) / sigma
            x = x + d * (sigma_next - sigma)
            
        # Zabezpieczenie przed przepełnieniem pamięci (overflow / NaN) w VAE
        x = x.clamp(-50.0, 50.0)
            
        x = x / self.scale_factor
        x_cf = self.vae.decode(x)
        return x_cf