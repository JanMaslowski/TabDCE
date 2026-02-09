from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch
from tabdce.dataset.dataset import TabularCounterfactualDataset

def validity(clf, x_cf_tensor: torch.Tensor, y_target: torch.Tensor) -> float:
    clf.eval()
    with torch.no_grad():
        logits = clf(x_cf_tensor)
        preds = torch.argmax(logits, dim=1)
        
    correct = (preds == y_target).float().sum().item()
    return correct / y_target.size(0)

def proximity_l2_scaled(x_orig_num: np.ndarray, x_cf_num: np.ndarray, ranges: np.ndarray) -> float:
    diff = (x_cf_num - x_orig_num) / ranges
    dist = np.linalg.norm(diff, axis=1)
    return float(dist.mean())

def sparsity_mixed(x_orig_num: np.ndarray, x_cf_num: np.ndarray, 
                   x_orig_cat: np.ndarray, x_cf_cat: np.ndarray, eps: float = 1e-3) -> float:
    N, D_num = x_orig_num.shape
    _, D_cat = x_orig_cat.shape
    D_total = D_num + D_cat
    
    diff_num = (np.abs(x_cf_num - x_orig_num) > eps).sum(axis=1)
    diff_cat = (x_orig_cat != x_cf_cat).sum(axis=1)
    
    total_changed = diff_num + diff_cat
    sparsity_num = (diff_num / D_num).mean() if D_num > 0 else 0.0
    sparsity_cat = (diff_cat / D_cat).mean() if D_cat > 0 else 0.0
    sparsity_mixed = float((total_changed / D_total).mean()) 
    return sparsity_num,sparsity_cat,sparsity_mixed

def collect_metrics(
    clf,
    dataset: TabularCounterfactualDataset,
    x_orig_tensor: torch.Tensor, 
    x_cf_tensor: torch.Tensor,
    y_target: torch.Tensor,
) -> Dict[str, float]:
    res = {}
    
    if clf is not None:
        res["validity"] = validity(clf, x_cf_tensor, y_target)
    else:
        res["validity"] = -1.0

    x_orig_raw = dataset.inverse_transform(x_orig_tensor)
    x_cf_raw = dataset.inverse_transform(x_cf_tensor)
    
    y_tgt_np = y_target.cpu().numpy()
    n_num = dataset.num_numerical
    
    x_orig_num = x_orig_raw[:, :n_num].astype(float)
    x_cf_num   = x_cf_raw[:, :n_num].astype(float)
    
    x_orig_cat = x_orig_raw[:, n_num:]
    x_cf_cat   = x_cf_raw[:, n_num:]

    if n_num > 0:
        mins = x_orig_num.min(axis=0, keepdims=True)
        maxs = x_orig_num.max(axis=0, keepdims=True)
        ranges = np.clip(maxs - mins, 1e-6, None)
        
        res["proximity_l2"] = proximity_l2_scaled(x_orig_num, x_cf_num, ranges)
    else:
        res["proximity_l2"] = 0.0

    res["sparsity_num"], res["sparsity_cat"], res["sparsity_mixed"] = sparsity_mixed(x_orig_num, x_cf_num, x_orig_cat, x_cf_cat)
    
    return res