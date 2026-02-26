import os
import yaml
import json
import argparse
import torch
import numpy as np
import wandb
import random

from tabdce.loops.train_tabsyn import train 
from tabdce.loops.train_classifier import train_classifier
from tabdce.dataset.dataset import TabularCounterfactualDataset, get_generic_data
from tabdce.utils.advanced_metrics import MetricsEvaluator


def load_config(path: str) -> dict:
    """Ładuje config jako czysty słownik."""
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
            ohe=train_dataset.ohe,       
            build_neighbors=False        
        )

    clf_model = None
    if 'classifier' in cfg and cfg['classifier'].get('train'):
        print("\n=== Training Classifier ===")
        clf_model = train_classifier(train_dataset, cfg['classifier'], device, val_dataset)
        clf_model.eval()

    print("\n=== Training Diffusion ===")
    trained_diffusion = train(cfg, train_dataset)
    trained_diffusion.eval()
    print("\n=== Generating Counterfactuals (Standard Sampling) ===")
    
    NUM_TEST = cfg['dataset'].get('n_test_samples', 500)
    N_CF_PER_SAMPLE = cfg['dataset'].get('n_cf_per_sample', 10) 
    
    indices = np.random.choice(len(X_test_raw), min(len(X_test_raw), NUM_TEST), replace=False)
    test_dataset_helper = TabularCounterfactualDataset(
        X=X_test_raw[indices], y=y_test[indices], spec=spec,
        device=device,
        scaler=train_dataset.scaler, 
        ohe=train_dataset.ohe,
        build_neighbors=False
    )
    
    x_orig = test_dataset_helper.X_model 
    y_test_tensor = test_dataset_helper.y
    y_target = torch.clamp((y_test_tensor + 1) % 2, 0, 1)

    x_input_expanded = x_orig.repeat_interleave(N_CF_PER_SAMPLE, dim=0)
    y_input_expanded = y_target.repeat_interleave(N_CF_PER_SAMPLE, dim=0)

    print(f"Test Samples: {len(x_orig)}")
    print(f"CF per Sample: {N_CF_PER_SAMPLE}")
    print(f"Total Batch Size: {x_input_expanded.shape[0]}")

    # --- SAMPLOWANIE (BEZ SVDD) ---
    with torch.no_grad():
        final_cfs = trained_diffusion.sample(
            x_orig=x_input_expanded,       
            y_target=y_input_expanded,     
            temperature=1.0,
            guidance_scale=10.0
        )
    
    group_ids = np.arange(len(x_orig)).repeat(N_CF_PER_SAMPLE)
    print("\n=== Calculating Metrics ===")
    evaluator = MetricsEvaluator(train_dataset)
    metrics = evaluator.evaluate(
        x_orig_tensor=x_input_expanded, 
        x_cf_tensor=final_cfs,     
        y_target_tensor=y_input_expanded,
        cf_group_ids=group_ids,
        X_train_np=train_dataset.inverse_transform(train_dataset.X_model),
        clf_model=clf_model  
    )
    
    for k, v in metrics.items():
        print(f"{k:<30}: {v:.4f}")

    wandb.log(metrics)
    x_orig_plot = train_dataset.inverse_transform(x_input_expanded)
    x_cf_plot = train_dataset.inverse_transform(final_cfs)

    save_path = os.path.join(cfg['train']['output_dir'], f"{cfg.get('project_name', 'run')}_metrics.pt")
    os.makedirs(cfg['train']['output_dir'], exist_ok=True)
    
    torch.save({
        "metrics": metrics, "config": cfg, 
        "model_state": trained_diffusion.state_dict(),
        "plot_data": {
            "x_orig": x_orig_plot,           
            "x_cf": x_cf_plot,               
            "y_target": y_input_expanded.cpu().numpy(),  
            "cf_group_ids": group_ids        
        }
    }, save_path)
    print(f"Saved to: {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()