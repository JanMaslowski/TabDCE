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
            # Słownik na akumulację wszystkich typów lossów w danej epoce
            epoch_losses = {}
            
            for batch in dataloader:
                x = batch["x_orig"].to(device)
                
                optimizer_vae.zero_grad()
                recon_num, recon_cats, mu, logvar = vae(x)
                
                # Zbieramy cały słownik z funkcji kosztu
                loss_dict = vae.loss_function(recon_num, recon_cats, x, mu, logvar)
                loss = loss_dict["loss"]
                
                loss.backward()
                optimizer_vae.step()
                
                # Akumulacja wszystkich metryk
                for k, v in loss_dict.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + val
                
            # Wyliczanie średnich dla epoki
            avg_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
            
            if (epoch + 1) % 10 == 0:
                # Dynamiczne generowanie logów dla konsoli
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                print(f"[VAE] Epoch {epoch+1}/{epochs_vae} | {loss_str}")
                
            # Logowanie wszystkich metryk do wandb (z prefiksem vae/)
            wandb_log_dict = {f"vae/{k}": v for k, v in avg_losses.items()}
            wandb_log_dict["epoch"] = epoch + 1
            wandb.log(wandb_log_dict)
            
        if vae_save_path:
            os.makedirs(os.path.dirname(vae_save_path), exist_ok=True)
            torch.save(vae.state_dict(), vae_save_path)
            print(f">>> VAE zapisane do: {vae_save_path}")

    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(">>> VAE wytrenowane i zamrożone.")
    
    # print("\n--- TEST VAE ---")
    # with torch.no_grad():
    #     test_batch = next(iter(dataloader))
    #     test_x = test_batch["x_orig"][:5].to(device) 
        
    #     # Test 1: Pełen przepływ (żeby sprawdzić, co widzi loss podczas treningu)
    #     recon_forward, _, _, _ = vae(test_x)
        
    #     # Test 2: Przejście przez encode/decode (tak, jak używa tego dyfuzja)
    #     latent = vae.encode(test_x)
    #     recon_x = vae.decode(latent)
        
    #     mse_forward = F.mse_loss(recon_forward, test_x).item()
    #     mse_diffusion = F.mse_loss(recon_x, test_x).item()
        
    #     print("Oryginał:\n", test_x.cpu().numpy().round(4))
    #     print("Rekonstrukcja (Encode->Decode):\n", recon_x.cpu().numpy().round(4))
    #     print(f"\nMSE w oryginalnym modelu (jak w treningu): {mse_forward:.6f}")
    #     print(f"MSE dla Dyfuzji (Encode->Decode): {mse_diffusion:.6f}")
    # print("----------------\n")
    
    diffusion_latent_dim = vae.flat_latent_dim
    print(f"\n=== FAZA 2: Trening Latent Diffusion ({epochs_diff} epok) ===")
    T = cfg['diffusion'].get('T', 200)
    
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
            
            # POPRAWKA: Obcinamy gradienty faktycznego obiektu dyfuzji!
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            
            optimizer_diff.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({
            "diffusion/loss": avg_loss,
            "epoch": epoch + 1 + epochs_vae # Epoki sumują się w wandb po treningu VAE
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"[Diff] Epoch {epoch+1}/{epochs_diff} | Loss: {avg_loss:.4f}")
            
    diffusion.eval()
    return diffusion