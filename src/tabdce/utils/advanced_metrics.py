import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import LocalOutlierFactor

class Metric:
    def __call__(self, **kwargs):
        raise NotImplementedError


class ProximityContinuousL1(Metric):

    def __call__(self, x_orig_tensor, x_cf_tensor, num_numerical, **kwargs):
        if num_numerical == 0: 
            return 0.0
        x_o = x_orig_tensor[:, :num_numerical]
        x_c = x_cf_tensor[:, :num_numerical]
        l1_dist = (x_o - x_c).abs().mean()
        return float(l1_dist.item())

class SparsityCategorical(Metric):

    def __call__(self, X_test_valid, X_cf_valid, categorical_features, **kwargs):
        if len(X_test_valid) == 0 or not categorical_features: 
            return 0.0
        
        D_cat = len(categorical_features)
        X_orig_cat = X_test_valid[:, categorical_features]
        X_cf_cat = X_cf_valid[:, categorical_features]
        diff_count = (X_orig_cat != X_cf_cat).sum(axis=1)
        return float(np.mean(diff_count / D_cat))

class EpsilonSparsity(Metric):

    def __call__(self, X_test_valid, X_cf_valid, continuous_features, ranges, **kwargs):
        if len(X_test_valid) == 0 or not continuous_features: 
            return 0.0
        
        epsilon = 0.05
        D_num = len(continuous_features)
        
        X_orig_cont = X_test_valid[:, continuous_features].astype(float)
        X_cf_cont = X_cf_valid[:, continuous_features].astype(float)
        abs_diff = np.abs(X_orig_cont - X_cf_cont)
        thresholds = epsilon * ranges.reshape(1, -1)
        significant_changes = (abs_diff > thresholds).astype(float)
        return float(np.mean(np.sum(significant_changes, axis=1) / D_num))


class DiversityMixed(Metric):

    def __call__(self, x_cf_tensor, x_cf_valid_raw, cf_group_ids_valid, num_numerical, categorical_features, **kwargs):
        if cf_group_ids_valid is None or len(x_cf_tensor) == 0: 
            return 0.0
        
        cf_group_ids = np.asarray(cf_group_ids_valid)
        unique_groups = np.unique(cf_group_ids)
        
        D_num = num_numerical
        D_cat = len(categorical_features)
        D_total = D_num + D_cat
        if D_total == 0: return 0.0

        group_diversities = []
        
        X_num = x_cf_tensor[:, :num_numerical].cpu().numpy()
        X_cat = x_cf_valid_raw[:, categorical_features]

        for group_id in unique_groups:
            indices = np.where(cf_group_ids == group_id)[0]
            if len(indices) < 2: 
                continue
            
            if D_num > 0:
                grp_num = X_num[indices]
                dists_num = pdist(grp_num, metric='cityblock')
            else:
                dists_num = 0.0

            if D_cat > 0:
                grp_cat = X_cat[indices]
                grp_cat_encoded = np.zeros(grp_cat.shape, dtype=int)
                for c in range(grp_cat.shape[1]):
                    _, encoded = np.unique(grp_cat[:, c], return_inverse=True)
                    grp_cat_encoded[:, c] = encoded
                
                dists_cat = pdist(grp_cat_encoded, metric='hamming') * D_cat
            else:
                dists_cat = 0.0
            
            mixed_dists = dists_num + dists_cat
            if mixed_dists.size > 0:
                group_diversities.append(np.mean(mixed_dists) / D_total)

        if not group_diversities: 
            return 0.0
            
        return float(np.mean(group_diversities))

class MetricsEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset 
        
        self.metrics = {
            "prox_cont": ProximityContinuousL1(),
            "spars_cat": SparsityCategorical(),
            "epsilon_spars": EpsilonSparsity(),
            "diversity": DiversityMixed()
        }

    def evaluate(self, 
                 x_orig_tensor: torch.Tensor, 
                 x_cf_tensor: torch.Tensor, 
                 y_target_tensor: torch.Tensor,
                 cf_group_ids: np.ndarray,
                 X_train_np: np.ndarray,
                 clf_model = None) -> dict:

        X_orig_np = self.dataset.inverse_transform(x_orig_tensor)
        X_cf_np = self.dataset.inverse_transform(x_cf_tensor)
        num_feat_idx = self.dataset.spec.num_idx
        cat_feat_idx = self.dataset.spec.cat_idx

        if clf_model is not None:
            clf_model.eval()
            with torch.no_grad():
                logits = clf_model(x_cf_tensor)
                preds = torch.argmax(logits, dim=1)
            valid_mask = (preds.cpu().numpy() == y_target_tensor.cpu().numpy())
            validity_score = valid_mask.mean()
        else:
            valid_mask = np.ones(len(X_cf_np), dtype=bool) 
            validity_score = 0.0

        X_orig_valid = X_orig_np[valid_mask]
        X_cf_valid = X_cf_np[valid_mask]
        group_ids_valid = cf_group_ids[valid_mask]
        valid_mask_tensor = torch.from_numpy(valid_mask).to(x_orig_tensor.device)
        x_orig_tensor_valid = x_orig_tensor[valid_mask_tensor]
        x_cf_tensor_valid = x_cf_tensor[valid_mask_tensor]

        results = {
            "validity": validity_score,
            "valid_count": int(valid_mask.sum()),
        }

        if len(X_cf_valid) == 0:
            return results

        if len(num_feat_idx) > 0:
            X_train_num = X_train_np[:, num_feat_idx].astype(float)
            mins = np.min(X_train_num, axis=0)
            maxs = np.max(X_train_num, axis=0)
            ranges = np.clip(maxs - mins, 1e-6, None)
        else:
            ranges = np.array([])

        kwargs = {
            "x_orig_tensor": x_orig_tensor_valid,
            "x_cf_tensor": x_cf_tensor_valid,
            "num_numerical": self.dataset.num_numerical,
            
            "X_test_valid": X_orig_valid,
            "X_cf_valid": X_cf_valid,
            "x_cf_valid_raw": X_cf_valid, 
            "categorical_features": cat_feat_idx,
            "continuous_features": num_feat_idx, 
            "ranges": ranges,
            "cf_group_ids_valid": group_ids_valid
        }

        print(f"Calculating metrics on {len(X_cf_valid)} valid CFs...")
        
        for name, metric_fn in self.metrics.items():
            try:
                val = metric_fn(**kwargs)
                results[name] = val
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                results[name] = 0.0

        return results