import os
import yaml
import argparse
import torch
import numpy as np
from types import SimpleNamespace
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from tabdce.loops.train import train
from tabdce.dataset.dataset import TabularCounterfactualDataset, TabularSpec


def load_config(path: str):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    def dict_to_ns(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_ns(v)
        return SimpleNamespace(**d)
    
    return dict_to_ns(cfg_dict)

def get_twomoons_data(n_samples: int, noise: float):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not torch.cuda.is_available() and cfg.train.device == "cuda":
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.train.device)

    train_cfg = SimpleNamespace(
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        epochs=cfg.train.epochs,
        T=cfg.diffusion.T,
        device=str(device),
        hidden_dim=cfg.model.hidden_dim
    )

    if cfg.dataset.name == "twomoons":
        X, y = get_twomoons_data(cfg.dataset.n_samples, cfg.dataset.noise)
        
        spec = TabularSpec(
            num_idx=[0, 1],
            cat_idx=[],
            cat_cardinalities=[]
        )
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dataset = TabularCounterfactualDataset(
                    X=X_train,
                    y=y_train,
                    spec=spec,
                    k=cfg.dataset.k_neighbors,
                    search_method=getattr(cfg.dataset, 'search_method', 'knn'),
                    device=device,
                    build_neighbors=True
                )
    
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} nie jest zaimplementowany.")

    print("Training started") 
    trained_diffusion = train(train_cfg, dataset)
    
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.train.output_dir, f"{cfg.project_name}_model.pt")
    
    torch.save({
        "model_state_dict": trained_diffusion.state_dict(),
        "config": cfg,
        "dataset_qt": dataset.qt,
        "dataset_ohe": dataset.ohe
    }, save_path)
    
    print(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()