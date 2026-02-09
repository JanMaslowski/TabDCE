import os
import yaml
import json
import argparse
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
import wandb

from tabdce.loops.train import train
from tabdce.loops.train_classifier import train_classifier
from tabdce.dataset.dataset import TabularCounterfactualDataset, TabularSpec, get_generic_data
from tabdce.utils.advanced_metrics import MetricsEvaluator

def load_yaml_config(path: str):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    def dict_to_ns(d):
        for k, v in d.items():
            if isinstance(v, dict): d[k] = dict_to_ns(v)
        return SimpleNamespace(**d)
    return dict_to_ns(cfg_dict), cfg_dict

def load_data_config(path: str):
    with open(path, 'r') as f: return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adult.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Brak pliku config.")
        return
    cfg, cfg_dict = load_yaml_config(args.config)

    wandb.init(
        project=getattr(cfg, "project_name", "tabdce-project"),
        name=getattr(cfg, "run_name", None), 
        config=cfg_dict 
    )
    device = torch.device(cfg.train.device if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    print(f"Device selected: {device}")
    data_dir = getattr(cfg.dataset, 'data_dir', 'data/adult')
    data_config = load_data_config(os.path.join(data_dir, "config.json"))
    X_train, X_test, y_train, y_test, spec, num_col_names = get_generic_data(data_dir, data_config)

    n_samples = getattr(cfg.dataset, 'n_samples', None)
    if n_samples and n_samples < len(y_train):
        idx = np.random.choice(len(y_train), n_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    dataset = TabularCounterfactualDataset(
        X=X_train, y=y_train, spec=spec, k=cfg.dataset.k_neighbors,
        search_method=getattr(cfg.dataset, 'search_method', 'knn'),
        device=device, build_neighbors=False 
    )

    clf_model = None
    if hasattr(cfg, 'classifier') and cfg.classifier.train:
        clf_cfg = SimpleNamespace(**vars(cfg.classifier))
        clf_model = train_classifier(dataset, clf_cfg, device)
        
        if getattr(cfg.classifier, 'relabel', False):
            print("RELABELING...")
            clf_model.eval()
            with torch.no_grad():
                dataset.y = torch.argmax(clf_model(dataset.X_model), dim=1).long()
                X_test_tensor = dataset.to_model_space(X_test).to(device)
                y_test = torch.argmax(clf_model(X_test_tensor), dim=1).cpu().numpy()
    
    dataset.neigh_idx = dataset._build_opposite_class_neighbors(
        dataset.X_model[:, :dataset.num_numerical], dataset.y, k=dataset.k
    )

    train_cfg = SimpleNamespace(
        batch_size=cfg.train.batch_size, lr=cfg.train.lr,
        epochs=cfg.train.epochs, T=cfg.diffusion.T,
        device=str(device), hidden_dim=getattr(cfg.model, 'hidden_dim', 128)
    )
    trained_diffusion = train(train_cfg, dataset)
    print("\n--- Rozpoczynanie zaawansowanej ewaluacji ---")
    NUM_TEST_INSTANCES = getattr(cfg.dataset, 'n_test_samples', 500)
    NUM_CF_PER_INSTANCE = getattr(cfg.dataset, 'n_cf_per_sample', 10)
    
    if len(X_test) > NUM_TEST_INSTANCES:
        test_indices = np.random.choice(len(X_test), NUM_TEST_INSTANCES, replace=False)
        X_test_sample = X_test[test_indices]
        y_test_sample = y_test[test_indices]
    else:
        X_test_sample = X_test
        y_test_sample = y_test

    x_orig_single = dataset.to_model_space(X_test_sample).to(device)
    y_target_single = torch.tensor(y_test_sample, device=device).long()
    x_orig_expanded = x_orig_single.repeat_interleave(NUM_CF_PER_INSTANCE, dim=0)
    y_target_expanded = y_target_single.repeat_interleave(NUM_CF_PER_INSTANCE, dim=0)
    
    group_ids = np.arange(len(x_orig_single)).repeat(NUM_CF_PER_INSTANCE)
    
    print(f"Generowanie {NUM_CF_PER_INSTANCE} CF dla {len(x_orig_single)} instancji.")
    print(f"Łączny rozmiar batcha generowania: {x_orig_expanded.shape[0]}")
    
    TEMPERATURE = 1.0
    print(f"Generowanie z temperaturą: {TEMPERATURE}")
    
    x_cf_expanded = trained_diffusion.sample_counterfactual(
        x_orig_expanded, 
        y_target_expanded,
        temperature=TEMPERATURE
    )
    
    evaluator = MetricsEvaluator(dataset)
    X_train_orig_np = dataset.inverse_transform(dataset.X_model)


    metrics = evaluator.evaluate(
        x_orig_tensor=x_orig_expanded,
        x_cf_tensor=x_cf_expanded,
        y_target_tensor=y_target_expanded,
        cf_group_ids=group_ids,
        X_train_np=X_train_orig_np,
        clf_model=clf_model  
    )
    
    print("\n=== WYNIKI KOŃCOWE ===")
    for k, v in metrics.items():
        print(f"{k:<30}: {v:.4f}")

    wandb.log(metrics)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.train.output_dir, f"{cfg.project_name}_metrics.pt")
    torch.save({
        "metrics": metrics,
        "config": cfg,
        "model_state": trained_diffusion.state_dict(),
        "clf_state": clf_model.state_dict() if clf_model else None
    }, save_path)
    print(f"Zapisano w: {save_path}")
    wandb.save(save_path)
    wandb.finish()
if __name__ == "__main__":
    main()