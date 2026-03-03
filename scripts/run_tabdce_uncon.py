import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import random
import pandas as pd

from tabdce.loops.train_uncondictioned import train 
from tabdce.loops.train_classifier import train_classifier
from tabdce.dataset.dataset import TabularCounterfactualDataset, get_generic_data
from tabdce.utils.advanced_metrics import MetricsEvaluator, generate_knn_counterfactuals

@torch.no_grad()
def sample_counterfactuals_tabdiff(
    diffusion_model, 
    x_num_orig: torch.Tensor, 
    x_cat_orig: torch.Tensor, 
    noise_level: float = 0.45, 
    temperature: float = 1.0
):
    device = diffusion_model.device
    b = x_num_orig.shape[0]
    
    t = torch.linspace(0, 1, diffusion_model.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion_model.num_schedule.total_noise(t)
    sigma_cat_cur = diffusion_model.cat_schedule.total_noise(t)
    
    sigma_num_next = torch.zeros_like(sigma_num_cur)
    sigma_num_next[1:] = sigma_num_cur[0:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur)
    sigma_cat_next[1:] = sigma_cat_cur[0:-1]

    start_step = int(noise_level * (diffusion_model.num_timesteps - 1))
    
    z_norm = x_num_orig + torch.randn_like(x_num_orig) * sigma_num_cur[start_step] * temperature
    
    if x_cat_orig.shape[1] > 0:
        move_chance = -torch.expm1(-sigma_cat_cur[start_step])
        z_cat, _ = diffusion_model.q_xt(x_cat_orig, move_chance.repeat(b, 1) if move_chance.dim() == 1 else move_chance)
    else:
        z_cat = x_cat_orig

    # Pętla EDM
    for i in reversed(range(0, start_step + 1)):
        z_norm, z_cat, _ = diffusion_model.edm_update(
            z_norm, z_cat, i, 
            t[i], t[i-1] if i > 0 else None, t[i],
            sigma_num_cur[i], sigma_num_next[i], sigma_num_cur[i], 
            sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_cur[i],
        )
        
    return z_norm, z_cat

def tabdiff_to_flat_tensor(z_norm, z_cat, cat_cardinalities):
    cat_parts = []
    if z_cat is not None and z_cat.shape[1] > 0:
        for i, card in enumerate(cat_cardinalities):
            oh = F.one_hot(z_cat[:, i].long(), num_classes=card).float()
            cat_parts.append(z_cat[:, i:i+1].float())
            
    if len(cat_parts) > 0:
        return torch.cat([z_norm] + cat_parts, dim=-1)
    return z_norm

class ClfFlatWrapper(nn.Module):
    def __init__(self, clf, num_num):
        super().__init__()
        self.clf = clf
        self.num_num = num_num
    def forward(self, x_flat):
        return self.clf(x_flat[:, :self.num_num], x_flat[:, self.num_num:])

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adult.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Missing config: {args.config}")
    
    cfg = load_config(args.config)
    seed_everything(seed = 42)
    wandb.init(
        project=cfg.get("project_name", "tabdce-project"),
        name=cfg.get("run_name", None), 
        config=cfg 
    )

    device_str = cfg['train'].get('device', 'cuda')
    if torch.cuda.is_available() and device_str == 'cuda':
        device = torch.device("cuda")
        print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Running on CPU")

    data_dir = cfg['dataset'].get('data_dir', 'data/adult')
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, spec = get_generic_data(data_dir, data_config)

    n_samples = cfg['dataset'].get('n_samples')
    if n_samples and n_samples < len(y_train):
        idx = np.random.choice(len(y_train), n_samples, replace=False)
        X_train_raw = X_train_raw[idx]
        y_train = y_train[idx]

    print("--- Building Train Dataset ---")
    train_dataset = TabularCounterfactualDataset(
        X=X_train_raw, y=y_train, spec=spec, 
        k=cfg['dataset']['k_neighbors'],
        search_method=cfg['dataset'].get('search_method', 'knn'),
        device=device,
        build_neighbors=True
    )

    val_dataset = None
    if X_val_raw is not None:
        print("--- Building Validation Dataset ---")
        val_dataset = TabularCounterfactualDataset(
            X=X_val_raw, y=y_val, spec=spec,
            k=cfg['dataset']['k_neighbors'],
            device=device,
            scaler=train_dataset.scaler, 
            ordinal_encoder=train_dataset.ordinal_encoder, # ZMIANA: ordinal_encoder zamiast OHE       
            build_neighbors=False        
        )

    clf_model = None
    clf_flat = None
    if 'classifier' in cfg and cfg['classifier'].get('train'):
        print("\n=== Training Classifier ===")
        clf_model = train_classifier(train_dataset, cfg['classifier'], device, val_dataset)
        clf_model.eval()
        # Wrapper dla Ewaluatora (który operuje na płaskim formacie)
        clf_flat = ClfFlatWrapper(clf_model, train_dataset.num_numerical)

    print("\n=== Training Diffusion (TabDiff) ===")
    trained_diffusion = train(cfg, train_dataset)
    trained_diffusion.eval()
    
    print("\n=== Generating Counterfactuals (SDEdit) ===")
    
    NUM_TEST = cfg['dataset'].get('n_test_samples', 500)
    N_CF_PER_SAMPLE = cfg['dataset'].get('n_cf_per_sample', 10) 
    
    indices = np.random.choice(len(X_test_raw), min(len(X_test_raw), NUM_TEST), replace=False)
    test_dataset_helper = TabularCounterfactualDataset(
        X=X_test_raw[indices], y=y_test[indices], spec=spec,
        device=device,
        scaler=train_dataset.scaler, 
        ordinal_encoder=train_dataset.ordinal_encoder, 
        build_neighbors=False
    )
    
    x_num_orig = test_dataset_helper.X_num
    x_cat_orig = test_dataset_helper.X_cat
    x_orig_flat = test_dataset_helper.X_model 
    
    y_test_tensor = test_dataset_helper.y
    y_target = torch.clamp((y_test_tensor + 1) % 2, 0, 1)
    x_num_expanded = x_num_orig.repeat_interleave(N_CF_PER_SAMPLE, dim=0)
    x_cat_expanded = x_cat_orig.repeat_interleave(N_CF_PER_SAMPLE, dim=0)
    x_flat_expanded = x_orig_flat.repeat_interleave(N_CF_PER_SAMPLE, dim=0)
    y_input_expanded = y_target.repeat_interleave(N_CF_PER_SAMPLE, dim=0)

    print(f"Test Samples: {len(x_orig_flat)}")
    print(f"CF per Sample: {N_CF_PER_SAMPLE}")
    print(f"Total Batch Size: {x_flat_expanded.shape[0]}")
    print("\n=== Generating Unconditional Samples ===")
    
    with torch.no_grad():
        generated_tensor = trained_diffusion.sample(num_samples=1000)
    generated_np = train_dataset.inverse_transform(generated_tensor)
    save_path = os.path.join(cfg['train']['output_dir'], "unconditional_samples.pt")
    torch.save({"generated_samples": generated_np}, save_path)
    
    print(f"✅ Zapisano bezwarunkowe próbki do: {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()