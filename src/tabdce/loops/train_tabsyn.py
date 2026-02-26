import os
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam

from tabdce.model.denoise_fn_tabsyn import TabularEpsModel
from tabdce.model.diffusion_tabsyn import LatentTabularDiffusion
from tabdce.model.vae import TabularVAE

def train(cfg: dict, dataset): 
    device_str = cfg['train'].get('device', 'cuda')
    
    if torch.cuda.is_available() and device_str == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = cfg['train'].get('batch_size', 128)
    batch_size = cfg['train'].get('batch_size', 128)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_dim = dataset.num_numerical
    cat_dims = dataset.cat_cardinalities 
    y_classes = max(2, dataset.num_classes_target)
    lr_vae     = cfg['vae'].get('lr', 1e-3)
    epochs_vae = cfg['vae'].get('epochs', 500)
    vae_path      = cfg['vae'].get('vae_path', None)
    vae_save_path = cfg['vae'].get('vae_save_path', None)
    
    lr_diff     = cfg['train'].get('lr', 1e-3)
    epochs_diff = cfg['train'].get('epochs', 1000)
    
    latent_dim = cfg['model'].get('latent_dim', 64)
    hidden_dim = cfg['model'].get('hidden_dim', 256)

    print(f"\n=== FAZA 1: Trening VAE ({epochs_vae} epok) ===")
    
    vae = TabularVAE(
        num_numerical=num_dim,
        cat_cardinalities=cat_dims,
        latent_dim=latent_dim,
        device=device
    ).to(device)
    
    if vae_path and os.path.exists(vae_path):
        print(f"\n=== FAZA 1: Wczytywanie VAE z pliku ({vae_path}) ===")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print(">>> VAE wczytane pomyślnie.")
    else:
        print(f"\n=== FAZA 1: Trening VAE od zera ({epochs_vae} epok) ===")
        if vae_path:
            print(f"⚠️ Uwaga: Plik vae_path '{vae_path}' nie istnieje. Wymuszono trening.")
            
        optimizer_vae = Adam(vae.parameters(), lr=lr_vae)
        vae.train()
        
        for epoch in range(epochs_vae):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch["x_orig"].to(device)
                
                optimizer_vae.zero_grad()
                recon_num, recon_cats, mu, logvar = vae(x)
                loss_dict = vae.loss_function(recon_num, recon_cats, x, mu, logvar)
                loss = loss_dict["loss"]
                
                loss.backward()
                optimizer_vae.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"[VAE] Epoch {epoch+1}/{epochs_vae} | Loss: {avg_loss:.4f}")
                
            wandb.log({"vae/loss": avg_loss, "epoch": epoch+1})
            
        if vae_save_path:
            os.makedirs(os.path.dirname(vae_save_path), exist_ok=True)
            torch.save(vae.state_dict(), vae_save_path)
            print(f">>> VAE zapisane do: {vae_save_path}")

    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(">>> VAE wytrenowane i zamrożone.")
    diffusion_latent_dim = vae.flat_latent_dim
    print(f"\n=== FAZA 2: Trening Latent Diffusion ({epochs_diff} epok) ===")
    T = cfg['diffusion'].get('T', 200)
    denoise_model = TabularEpsModel(
        latent_dim=diffusion_latent_dim, 
        y_classes=y_classes,
        hidden=hidden_dim
    ).to(device)
    
    diffusion = LatentTabularDiffusion(
        denoise_fn=None,
        vae_model=vae,
        latent_dim=diffusion_latent_dim,
        T=T,
        device=device
    ).to(device)

    optimizer_diff = Adam(diffusion.parameters(), lr=lr_diff)
    diffusion.train()
    
    for epoch in range(epochs_diff):
        epoch_loss = 0.0
        for batch in dataloader:
            x_orig = batch["x_orig"].to(device)
            y_tgt = batch["y_target"].to(device)
            x_neigh = batch["x_neigh"].to(device)
            
            optimizer_diff.zero_grad()
            loss = diffusion(x_neigh, x_orig, y_tgt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), max_norm=1.0)
            optimizer_diff.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({
            "diffusion/loss": avg_loss,
            "epoch": epoch + 1 + epochs_vae
        })
        print(f"[Diff] Epoch {epoch+1}/{epochs_diff} | Loss: {avg_loss:.4f}")
    diffusion.eval()
    return diffusion