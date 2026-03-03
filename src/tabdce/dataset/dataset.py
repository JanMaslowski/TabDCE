from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OrdinalEncoder

@dataclass
class TabularSpec:
    num_idx: List[int]
    cat_idx: List[int]
    target_col: str = None 
    cat_cardinalities: List[int] = None

class TabularCounterfactualDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        spec: TabularSpec,
        k: int = 15,
        search_method: Literal["knn", "dpp"] = "knn",
        dpp_pool_factor: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        scaler_type="quantile",
        scaler: object | None = None,
        ordinal_encoder: OrdinalEncoder | None = None,
        build_neighbors: bool = True,
        gower_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.spec = spec
        self.k = k
        self.search_method = search_method
        self.dpp_pool_factor = dpp_pool_factor
        self.gower_weight = gower_weight

        self.y = y.long().to(device) if isinstance(y, torch.Tensor) else torch.from_numpy(y).long().to(device)
        self.num_classes_target = len(torch.unique(self.y))
        
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        N = X_np.shape[0]

        self.num_numerical = len(spec.num_idx)
        if self.num_numerical > 0:
            X_num_np = X_np[:, spec.num_idx].astype(np.float32)
            if scaler is not None:
                self.scaler = scaler
                X_num_tr = self.scaler.transform(X_num_np)
            else:
                self.scaler = StandardScaler() if scaler_type == "standard" else QuantileTransformer(output_distribution='normal')
                X_num_tr = self.scaler.fit_transform(X_num_np)
        else:
            self.scaler = None
            X_num_tr = np.zeros((N, 0), dtype=np.float32)

        if len(spec.cat_idx) > 0:
            X_cat_np = X_np[:, spec.cat_idx]
            if ordinal_encoder is None:
                self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_cat_idx = self.ordinal_encoder.fit_transform(X_cat_np).astype(np.int64)
            else:
                self.ordinal_encoder = ordinal_encoder
                X_cat_idx = self.ordinal_encoder.transform(X_cat_np).astype(np.int64)
            self.cat_cardinalities = [len(cats) for cats in self.ordinal_encoder.categories_]
        else:
            self.ordinal_encoder = None
            self.cat_cardinalities = []
            X_cat_idx = np.zeros((N, 0), dtype=np.int64)

        if spec.cat_cardinalities is None:
            spec.cat_cardinalities = self.cat_cardinalities

        self.X_num = torch.from_numpy(X_num_tr.astype(np.float32)).to(device)
        self.X_cat = torch.from_numpy(X_cat_idx).to(device)
        X_model_np = np.concatenate([X_num_tr, X_cat_idx.astype(np.float32)], axis=1)
        self.X_model = torch.from_numpy(X_model_np).to(device).to(dtype)

        self.neigh_idx = self._build_opposite_class_neighbors(self.X_model, self.y, k=self.k) if build_neighbors else None


    def _build_opposite_class_neighbors(self, X_model: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
        N = X_model.size(0)
        neigh_all = torch.zeros((N, k), dtype=torch.long, device=self.device)
        classes = y.unique().tolist()
        k_pool = k * self.dpp_pool_factor if self.search_method == "dpp" else k

        with torch.no_grad():
            for cls in classes:
                src_idx = (y == cls).nonzero(as_tuple=False).squeeze(1)
                tgt_idx = (y != cls).nonzero(as_tuple=False).squeeze(1)

                if src_idx.numel() == 0 or tgt_idx.numel() == 0: continue
                A = X_model[src_idx]
                B = X_model[tgt_idx]
                
                dists = torch.cdist(A[:, :self.num_numerical], B[:, :self.num_numerical], p=1) / max(1, self.num_numerical)
                
                if self.X_cat.shape[1] > 0:
                    cat_A = A[:, self.num_numerical:]
                    cat_B = B[:, self.num_numerical:]
                    cat_diffs = (cat_A.unsqueeze(1) != cat_B.unsqueeze(0)).float().mean(dim=-1)
                    dists += cat_diffs * self.gower_weight

                curr_pool = min(k_pool, int(tgt_idx.size(0)))
                _, topk_local_idx = torch.topk(dists, k=curr_pool, largest=False, sorted=True)
                candidates_global = tgt_idx[topk_local_idx]

                if self.search_method == "knn":
                    chosen = candidates_global[:, :k]
                elif self.search_method == "dpp":
                    chosen = self._select_dpp_greedy_fast(X_model[src_idx], X_model[candidates_global], candidates_global, k)
                if chosen.size(1) < k:
                    chosen = torch.cat([chosen, chosen[:, -1:].repeat(1, k - chosen.size(1))], dim=1)

                neigh_all[src_idx] = chosen

        return neigh_all
    
    def _select_dpp_greedy_fast(self, query_feats, cand_feats, cand_indices, k):
        B, Pool, F = cand_feats.shape
        if Pool <= k: return cand_indices
        sigma_q, sigma_s = 1.0, 5.0
        device = cand_feats.device
        dist_qc = torch.cdist(query_feats.unsqueeze(1), cand_feats).squeeze(1) ** 2
        Q = torch.exp(-dist_qc / sigma_q)
        norm_c = (cand_feats ** 2).sum(dim=2, keepdim=True)
        dist_cc = norm_c + norm_c.transpose(1, 2) - 2 * torch.bmm(cand_feats, cand_feats.transpose(1, 2))
        S = torch.exp(-dist_cc / sigma_s) + torch.eye(Pool, device=device).unsqueeze(0) * 1e-4
        L = S * (Q.unsqueeze(2) * Q.unsqueeze(1))
        
        selected_indices_local = torch.zeros((B, k), dtype=torch.long, device=device)
        K = L.clone()
        mask = torch.zeros((B, Pool), dtype=torch.bool, device=device)

        for step in range(k):
            diags = torch.diagonal(K, dim1=1, dim2=2)
            gains = diags.clone()
            gains[mask] = -float('inf')
            best_idx = torch.argmax(gains, dim=1)
            selected_indices_local[:, step] = best_idx
            mask.scatter_(1, best_idx.unsqueeze(1), True)
            if step < k - 1:
                best_idx_view = best_idx.view(B, 1, 1).expand(B, Pool, 1)
                v = K.gather(2, best_idx_view)
                d = v.gather(1, best_idx.view(B, 1, 1)).squeeze(2)
                K = K - torch.bmm(v, v.transpose(1, 2)) / (d.unsqueeze(2) + 1e-6)
        return torch.gather(cand_indices, 1, selected_indices_local)

    def __len__(self) -> int:
        return self.X_num.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_num_orig = self.X_num[idx]
        x_cat_orig = self.X_cat[idx]
        y_orig = self.y[idx]
        
        if self.neigh_idx is not None:
            cand_indices = self.neigh_idx[idx]
            chosen_idx = cand_indices[torch.randint(0, cand_indices.numel(), (1,)).item()]
            
            x_num_neigh = self.X_num[chosen_idx]
            x_cat_neigh = self.X_cat[chosen_idx]
            y_tgt = self.y[chosen_idx]
        else:
            x_num_neigh, x_cat_neigh, y_tgt = x_num_orig.clone(), x_cat_orig.clone(), y_orig
            
        return {
            "x_num_orig": x_num_orig, 
            "x_cat_orig": x_cat_orig,
            "y": y_orig, 
            "x_num_neigh": x_num_neigh, 
            "x_cat_neigh": x_cat_neigh,
            "y_target": y_tgt
        }

    def inverse_transform(self, x_model: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x_model, torch.Tensor):
            x_model = x_model.detach().cpu().numpy()
            
        N = x_model.shape[0]
        if self.num_numerical > 0:
            x_num_tr = np.clip(x_model[:, :self.num_numerical], -5.2, 5.2) 
            x_num_orig = self.scaler.inverse_transform(x_num_tr)
        else:
            x_num_orig = np.zeros((N, 0))
            
        if self.ordinal_encoder is not None:
            x_cat_idx = x_model[:, self.num_numerical:]
            x_cat_idx = np.round(x_cat_idx).astype(int)
            for i, card in enumerate(self.cat_cardinalities):
                x_cat_idx[:, i] = np.clip(x_cat_idx[:, i], 0, card - 1)
                
            x_cat_orig = self.ordinal_encoder.inverse_transform(x_cat_idx)
        else:
            x_cat_orig = np.zeros((N, 0))
            
        return np.concatenate([x_num_orig, x_cat_orig], axis=1)
    
    def relabel(self, clf_model, device, batch_size=256):
        loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        all_preds = []
        
        clf_model.eval()
        with torch.no_grad():
            for batch in loader:
                x_num = batch["x_num_orig"].to(device)
                x_cat = batch["x_cat_orig"].to(device)
                
                outputs = clf_model(x_num, x_cat)
                if outputs.dim() == 1 or outputs.shape[-1] == 1:
                    preds = (outputs.squeeze() > 0.5).float()
                else:
                    preds = outputs.argmax(dim=1).float()
                    
                all_preds.append(preds.cpu())
                
        self.y = torch.cat(all_preds, dim=0)
        
        counts = torch.bincount(self.y.long())
        print(f"[Dataset]  New class distribution: {counts.tolist()}")
        if hasattr(self, 'k') and getattr(self, 'build_neighbors', False):
            self._build_opposite_class_neighbors()

def get_generic_data(data_dir: str, data_config: dict):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    val_path = os.path.join(data_dir, "val.csv")
    
    print(f"Loading data from {data_dir}...")
    df_train = pd.read_csv(train_path, skipinitialspace=True)
    df_test = pd.read_csv(test_path, skipinitialspace=True)
    
    has_val = os.path.exists(val_path)
    if has_val:
        print("Found val.csv")
        df_val = pd.read_csv(val_path, skipinitialspace=True)
    else:
        print("No val.csv found. Will return None for X_val, y_val.")
        df_val = pd.DataFrame()

    df_train['split'] = 'train'
    df_test['split'] = 'test'
    if has_val:
        df_val['split'] = 'val'
        df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    else:
        df_full = pd.concat([df_train, df_test], ignore_index=True)
    num_cols = data_config.get("numerical_columns", [])
    cat_cols = data_config.get("categorical_columns", [])
    target_col = data_config.get("target_column")
    positive_val = data_config.get("target_positive_value") 
    missing_vals = data_config.get("missing_values", ["nan", "NaN", "?"])

    df_full[target_col] = df_full[target_col].astype(str).str.strip().str.replace('.', '', regex=False)
    if positive_val:
        df_full[target_col] = (df_full[target_col] == str(positive_val)).astype(int)
    else:
        df_full[target_col] = pd.to_numeric(df_full[target_col], errors='coerce').fillna(0).astype(int)

    y_full = df_full[target_col].to_numpy()
    for col in num_cols:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
            df_full[col] = df_full[col].fillna(df_full[col].median())

    for col in cat_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].astype(str).str.strip()
            df_full[col] = df_full[col].replace(missing_vals, np.nan)
            if df_full[col].isnull().all():
                df_full[col] = "Missing"
            else:
                mode_val = df_full[col].mode()[0]
                df_full[col] = df_full[col].fillna(mode_val)
    
    valid_num = [c for c in num_cols if c in df_full.columns]
    valid_cat = [c for c in cat_cols if c in df_full.columns]

    X_num = df_full[valid_num].to_numpy().astype(np.float32)
    X_cat = df_full[valid_cat].to_numpy()
    X_final = np.concatenate([X_num, X_cat], axis=1)

    spec = TabularSpec(
        num_idx=list(range(len(valid_num))),
        cat_idx=list(range(len(valid_num), len(valid_num) + len(valid_cat))),
        target_col=target_col
    )
    
    mask_train = (df_full['split'] == 'train').values
    mask_test = (df_full['split'] == 'test').values
    
    X_train, y_train = X_final[mask_train], y_full[mask_train]
    X_test, y_test = X_final[mask_test], y_full[mask_test]
    
    if has_val:
        mask_val = (df_full['split'] == 'val').values
        X_val, y_val = X_final[mask_val], y_full[mask_val]
    else:
        X_val, y_val = None, None

    return X_train, X_val, X_test, y_train, y_val, y_test, spec