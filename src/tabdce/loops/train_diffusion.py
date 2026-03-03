import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from tabdce.model.modules import UniModMLP, Model
from tabdce.model.diffusion import UnifiedCtimeDiffusion

class ConditionalDenoiser(nn.Module):
    def __init__(self, base_model: nn.Module, num_num: int, sum_cat_t: int, drop_prob: float = 0.25):
        super().__init__()
        self.base_model = base_model
        self.num_num = num_num
        self.sum_cat_t = sum_cat_t
        self.drop_prob = drop_prob
        
        self.cond_num = None
        self.cond_cat_soft = None
        self.cond_y_soft = None

    def set_condition(self, cond_num, cond_cat_soft, cond_y_soft):
        self.cond_num = cond_num
        self.cond_cat_soft = cond_cat_soft
        self.cond_y_soft = cond_y_soft

    def forward(self, x_num_t, x_cat_t_soft, t, sigma=None):
        apply_drop = self.training and (torch.rand(1).item() < self.drop_prob)
        
        if apply_drop:
            cond_num_used = torch.zeros_like(self.cond_num) if self.cond_num is not None else None
            cond_cat_used = torch.zeros_like(self.cond_cat_soft) if self.cond_cat_soft is not None else None
        else:
            cond_num_used = self.cond_num
            cond_cat_used = self.cond_cat_soft

        x_num_comb = torch.cat([x_num_t, cond_num_used], dim=1) if cond_num_used is not None else x_num_t
        cat_parts = [x_cat_t_soft]
        if cond_cat_used is not None and cond_cat_used.shape[1] > 0:
            cat_parts.append(cond_cat_used)
        if self.cond_y_soft is not None:
            cat_parts.append(self.cond_y_soft)
            
        x_cat_comb = torch.cat(cat_parts, dim=1) if len(cat_parts) > 1 else x_cat_t_soft
        
        out_num_comb, out_cat_comb = self.base_model(x_num_comb, x_cat_comb, t)
        
        return out_num_comb[:, :self.num_num], out_cat_comb[:, :self.sum_cat_t]

    

def train(cfg: dict, dataset): 
    device_str = cfg['train'].get('device', 'cuda')
    device = torch.device("cuda") if torch.cuda.is_available() and device_str == 'cuda' else torch.device("cpu")

    batch_size = cfg['train'].get('batch_size', 128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_dim = dataset.num_numerical
    cat_dims = dataset.cat_cardinalities 
    y_classes = max(2, dataset.num_classes_target)
    
    cat_dims_w_mask = [c + 1 for c in cat_dims]
    d_numerical_aug = num_dim * 2
    categories_aug = cat_dims_w_mask + cat_dims_w_mask + [y_classes]
    
    unimod = UniModMLP(
        d_numerical=d_numerical_aug,
        categories=categories_aug,
        num_layers=cfg['model'].get('num_layers', 4),
        d_token=cfg['model'].get('d_token', 64),
        n_head=cfg['model'].get('n_head', 1),
        factor=cfg['model'].get('factor', 4),
        bias=True,
        dim_t=cfg['model'].get('dim_t', 256),
        use_mlp=True
    )
    
    sum_cat_t = sum(cat_dims_w_mask)
    cond_denoiser = ConditionalDenoiser(unimod, num_dim, sum_cat_t)
    
    denoise_fn = Model(
        denoise_fn=cond_denoiser,
        sigma_data=1.0,
        precond=True,
        net_conditioning="sigma"
    )

    T = cfg['diffusion'].get('T', 100)
    epochs_diff = cfg['train'].get('epochs', 1000)
    lr_diff = cfg['train'].get('lr', 1e-3)
    
    print(f"\n=== FAZA: Trening TabDiff ({epochs_diff} epok) ===")
    
    tabdiff = UnifiedCtimeDiffusion(
        num_classes=np.array(cat_dims),
        num_numerical_features=num_dim,
        denoise_fn=denoise_fn,
        y_only_model=None, 
        num_timesteps=T,
        scheduler='power_mean_per_column',
        cat_scheduler='log_linear_per_column',
        noise_dist='uniform_t',
        edm_params={'sigma_data': 0.5},
        sampler_params={
            'stochastic_sampler': True, 
            'second_order_correction': True
        },
        device=device
    ).to(device)

    optimizer_diff = Adam(tabdiff.parameters(), lr=lr_diff)
    scheduler = CosineAnnealingLR(optimizer_diff, T_max=epochs_diff, eta_min=1e-5)
    tabdiff.train()
    
    for epoch in range(epochs_diff):
        epoch_loss = 0.0
        
        for batch in dataloader:
            x_num_orig = batch["x_num_orig"].to(device)
            x_cat_orig = batch["x_cat_orig"].to(device)
            y_tgt = batch["y_target"].to(device)
            
            x_num_neigh = batch["x_num_neigh"].to(device)
            x_cat_neigh = batch["x_cat_neigh"].to(device)
            
            cond_cat_parts = []
            if x_cat_orig.shape[1] > 0:
                for i, card in enumerate(dataset.cat_cardinalities):
                    oh = F.one_hot(x_cat_orig[:, i].long(), num_classes=card+1).float()
                    cond_cat_parts.append(oh)
                cond_cat_soft = torch.cat(cond_cat_parts, dim=-1)
            else:
                cond_cat_soft = torch.empty(x_num_orig.shape[0], 0, device=device)
                
            cond_y_soft = F.one_hot(y_tgt.long(), num_classes=y_classes).float()
            cond_denoiser.set_condition(x_num_orig, cond_cat_soft, cond_y_soft)
            x_target = torch.cat([x_num_neigh, x_cat_neigh.float()], dim=1)
            optimizer_diff.zero_grad()
            
            d_loss, c_loss = tabdiff.mixed_loss(x_target)
            loss = d_loss + c_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tabdiff.parameters(), max_norm=1.0)
            optimizer_diff.step()
            
            epoch_loss += loss.item()
        scheduler.step()    
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer_diff.param_groups[0]['lr']
        wandb.log({
            "tabdiff/loss": avg_loss,
            "epoch": epoch + 1 ,
            "tabdiff/lr": current_lr
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"[TabDiff] Epoch {epoch+1}/{epochs_diff} | Loss: {avg_loss:.4f}")
            
    tabdiff.eval()
    tabdiff.set_condition = cond_denoiser.set_condition
    
    return tabdiff