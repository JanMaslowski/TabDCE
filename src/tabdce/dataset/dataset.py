from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler

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
        ohe: OneHotEncoder | None = None,
        scaler: object | None = None,
        build_neighbors: bool = True,
        gower_weight: float = 1.0,
        data_dir: str = ""
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.spec = spec
        self.k = k
        self.search_method = search_method
        self.dpp_pool_factor = dpp_pool_factor
        self.scaler_type = scaler_type
        self.gower_weight = gower_weight
        self.data_dir = data_dir

        if isinstance(y, torch.Tensor):
            self.y = y.long().to(self.device)
        else:
            self.y = torch.from_numpy(y).long().to(self.device)
        
        self.num_classes_target = len(torch.unique(self.y))
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        N = X_np.shape[0]
        if len(spec.num_idx) > 0:
            X_num_np = X_np[:, spec.num_idx].astype(np.float32)
        else:
            X_num_np = np.zeros((N, 0), dtype=np.float32)

        if len(spec.cat_idx) > 0:
            X_cat_np = X_np[:, spec.cat_idx]
        else:
            X_cat_np = np.zeros((N, 0))

        self.num_numerical = X_num_np.shape[1]
        if self.num_numerical > 0:
            if scaler is not None:
                print("Using provided scaler (transform only).")
                self.scaler = scaler
                X_num_tr = self.scaler.transform(X_num_np)
            else:
                if self.scaler_type == "standard":
                    print("Fitting new StandardScaler...")
                    self.scaler = StandardScaler()
                else:
                    print("Fitting new QuantileTransformer...")
                    self.scaler = QuantileTransformer(output_distribution='normal')
                
                X_num_tr = self.scaler.fit_transform(X_num_np)
        else:
            self.scaler = None
            X_num_tr = np.zeros((N, 0), dtype=np.float32)

        if X_cat_np.shape[1] > 0:
            if ohe is None:
                print("Fitting new OneHotEncoder...")
                self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_cat_oh = self.ohe.fit_transform(X_cat_np)
            else:
                print("Using provided OneHotEncoder (transform only).")
                self.ohe = ohe
                X_cat_oh = self.ohe.transform(X_cat_np)
            
            self.cat_cardinalities = [len(c) for c in self.ohe.categories_]
            X_cat_log = np.log(np.clip(X_cat_oh, 1e-30, 1.0)).astype(np.float32)
        else:
            self.ohe = None
            self.cat_cardinalities = []
            X_cat_log = np.zeros((N, 0), dtype=np.float32)

        if spec.cat_cardinalities is None:
            spec.cat_cardinalities = self.cat_cardinalities

        X_model_np = np.concatenate([X_num_tr.astype(np.float32), X_cat_log], axis=1)
        self.X_model = torch.from_numpy(X_model_np).to(self.device).to(dtype)
        if build_neighbors:
            if self.num_numerical > 0:
                X_feat = torch.from_numpy(X_num_tr.astype(np.float32)).to(self.device)
            else:
                X_feat = torch.zeros((N, 1), device=self.device)
            
            print(f"Building neighbors using method: {self.search_method.upper()}")
            self.neigh_idx = self._build_opposite_class_neighbors(X_feat, self.y, k=self.k)
        else:
            self.neigh_idx = None

    def _build_opposite_class_neighbors(self, X_feat: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
        N = self.X_model.size(0)
        neigh_all = torch.zeros((N, k), dtype=torch.long, device=self.device)
        classes = y.unique().tolist()
        
        k_pool = k * self.dpp_pool_factor if self.search_method == "dpp" else k
        if self.num_numerical > 0:
            X_num_all = self.X_model[:, :self.num_numerical]
        else:
            X_num_all = None

        n_cat_cols_ohe = self.X_model.shape[1] - self.num_numerical
        if n_cat_cols_ohe > 0:
            X_cat_all = (self.X_model[:, self.num_numerical:] > -0.5).float()
            n_cat_features_orig = len(self.spec.cat_idx)
        else:
            X_cat_all = None
            n_cat_features_orig = 0

        with torch.no_grad():
            for cls in classes:
                src_mask = (y == cls)
                tgt_mask = (y != cls)
                
                src_idx = src_mask.nonzero(as_tuple=False).squeeze(1)
                tgt_idx = tgt_mask.nonzero(as_tuple=False).squeeze(1)

                if src_idx.numel() == 0: continue
                if tgt_idx.numel() == 0:
                    rand = torch.randint(0, N, (src_idx.numel(), k), device=self.device)
                    neigh_all[src_idx] = rand
                    continue
                dists = torch.zeros((src_idx.numel(), tgt_idx.numel()), device=self.device)
                
                if X_num_all is not None:
                    A_num = X_num_all[src_idx]
                    B_num = X_num_all[tgt_idx]
                    dist_num = torch.cdist(A_num, B_num, p=1)
                    dists += dist_num / self.num_numerical

                if X_cat_all is not None and n_cat_features_orig > 0:
                    A_cat = X_cat_all[src_idx]
                    B_cat = X_cat_all[tgt_idx]
                    matches = torch.mm(A_cat, B_cat.t())
                    dist_cat = 1.0 - (matches / n_cat_features_orig)
                    
                    dists += dist_cat * self.gower_weight

                curr_pool = min(k_pool, int(tgt_idx.size(0)))
                _, topk_local_idx = torch.topk(dists, k=curr_pool, largest=False, sorted=True)
                candidates_global = tgt_idx[topk_local_idx]

                if self.search_method == "knn":
                    chosen = candidates_global[:, :k]
                
                elif self.search_method == "dpp":
                    query_feats = self.X_model[src_idx]
                    cand_feats = self.X_model[candidates_global] 
                    chosen = self._select_dpp_greedy_fast(
                        query_feats=query_feats,
                        cand_feats=cand_feats,
                        cand_indices=candidates_global,
                        k=k
                    )

                if chosen.size(1) < k:
                    pad_size = k - chosen.size(1)
                    pad = chosen[:, -1:].repeat(1, pad_size)
                    chosen = torch.cat([chosen, pad], dim=1)

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
        return self.X_model.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_orig = self.X_model[idx]
        y_orig = self.y[idx]
        if self.neigh_idx is not None:
            cand_indices = self.neigh_idx[idx]
            chosen_idx = cand_indices[torch.randint(0, cand_indices.numel(), (1,)).item()]
            x_neigh = self.X_model[chosen_idx]
            y_tgt = self.y[chosen_idx]
        else:
            x_neigh = x_orig.clone()
            y_tgt = y_orig 
        return {"x_orig": x_orig, "y": y_orig, "x_neigh": x_neigh, "y_target": y_tgt}

    def to_model_space(self, X_raw: np.ndarray | pd.DataFrame) -> torch.Tensor:
        if isinstance(X_raw, pd.DataFrame):
            X_raw = X_raw.values

        N = X_raw.shape[0]
        if self.num_numerical > 0:
            X_num = X_raw[:, self.spec.num_idx].astype(np.float32)
            X_num_tr = self.scaler.transform(X_num)
        else:
            X_num_tr = np.zeros((N, 0), dtype=np.float32)
        if self.ohe is not None:
            X_cat = X_raw[:, self.spec.cat_idx]
            X_cat_oh = self.ohe.transform(X_cat)
            X_cat_log = np.log(np.clip(X_cat_oh, 1e-30, 1.0)).astype(np.float32)
        else:
            X_cat_log = np.zeros((N, 0), dtype=np.float32)
        X_model = np.concatenate([X_num_tr, X_cat_log], axis=1)
        return torch.from_numpy(X_model).to(self.device).to(self.dtype)

    def inverse_transform(self, x_model: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x_model, torch.Tensor):
            x_model = x_model.detach().cpu().numpy()
        N = x_model.shape[0]
        if self.num_numerical > 0:
            x_num_tr = x_model[:, :self.num_numerical]
            x_num_tr = np.clip(x_num_tr, -5.2, 5.2) 
            x_num_orig = self.scaler.inverse_transform(x_num_tr)
        else:
            x_num_orig = np.zeros((N, 0))
        if self.ohe is not None:
            x_cat_part = x_model[:, self.num_numerical:]
            indices_list = []
            start = 0
            for k in self.cat_cardinalities:
                part = x_cat_part[:, start:start+k]
                idx = np.argmax(part, axis=1)
                indices_list.append(idx.reshape(-1, 1))
                start += k
            x_cat_orig_list = []
            cat_indices = np.hstack(indices_list)
            for i, cats in enumerate(self.ohe.categories_):
                col_indices = cat_indices[:, i]
                orig_vals = cats[col_indices]
                x_cat_orig_list.append(orig_vals.reshape(-1, 1))
            x_cat_orig = np.hstack(x_cat_orig_list)
        else:
            x_cat_orig = np.zeros((N, 0))
        return np.concatenate([x_num_orig, x_cat_orig], axis=1)


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