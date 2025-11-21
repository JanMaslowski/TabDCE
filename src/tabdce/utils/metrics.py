# src/utils/metrics.py

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch
from tabdce.dataset.dataset import TabularCounterfactualDataset

def validity(clf, x_cf_clf_ready: np.ndarray, y_target: np.ndarray) -> float:
    y_pred = clf.predict(x_cf_clf_ready)
    return float((y_pred == y_target).mean())

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
    return float((total_changed / D_total).mean())

def collect_metrics(
    clf,
    dataset: TabularCounterfactualDataset,
    x_orig_tensor: torch.Tensor, # [B, D_model] - oryginalne
    x_cf_tensor: torch.Tensor,   # [B, D_model] - wygenerowane CF
    y_target: torch.Tensor,      # [B]
) -> Dict[str, float]:
    

    x_orig_raw = dataset.inverse_transform(x_orig_tensor)
    x_cf_raw = dataset.inverse_transform(x_cf_tensor)
    
    y_tgt_np = y_target.cpu().numpy()
    n_num = dataset.num_numerical
    
    x_orig_num = x_orig_raw[:, :n_num].astype(float)
    x_cf_num   = x_cf_raw[:, :n_num].astype(float)
    
    x_orig_cat = x_orig_raw[:, n_num:]
    x_cf_cat   = x_cf_raw[:, n_num:]

    if dataset.qt is not None:
        x_cf_clf_num = dataset.qt.transform(x_cf_num)
    else:
        x_cf_clf_num = x_cf_num

    if dataset.ohe is not None:
        x_cf_clf_cat = dataset.ohe.transform(x_cf_cat)
    else:
        x_cf_clf_cat = np.zeros((x_cf_raw.shape[0], 0))
        
    x_cf_for_clf = np.concatenate([x_cf_clf_num, x_cf_clf_cat], axis=1)


    res = {}
    

    res["validity"] = validity(clf, x_cf_for_clf, y_tgt_np)
    
    if n_num > 0:

        mins = x_orig_num.min(axis=0, keepdims=True)
        maxs = x_orig_num.max(axis=0, keepdims=True)
        ranges = np.clip(maxs - mins, 1e-6, None)
        
        res["proximity_l2"] = proximity_l2_scaled(x_orig_num, x_cf_num, ranges)
    else:
        res["proximity_l2"] = 0.0

    res["sparsity"] = sparsity_mixed(x_orig_num, x_cf_num, x_orig_cat, x_cf_cat)
    
    return res