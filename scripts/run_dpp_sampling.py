import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tabdce.loops.train_classifier import train_classifier
from tabdce.dataset.dataset import TabularCounterfactualDataset, get_generic_data
from tabdce.utils.advanced_metrics import MetricsEvaluator

from tabdce.model.modules import UniModMLP, Model
from tabdce.model.diffusion import UnifiedCtimeDiffusion


class ConditionalDenoiser(nn.Module):
    def __init__(self, base_model, num_num, sum_cat_t, drop_prob=0.25):
        super().__init__()
        self.base_model = base_model
        self.num_num = num_num
        self.sum_cat_t = sum_cat_t
        self.drop_prob = drop_prob
        self.cond_num = self.cond_cat_soft = self.cond_y_soft = None

    def set_condition(self, cond_num, cond_cat_soft, cond_y_soft):
        self.cond_num = cond_num
        self.cond_cat_soft = cond_cat_soft
        self.cond_y_soft = cond_y_soft

    def forward(self, x_num_t, x_cat_t_soft, t, sigma=None):
        cond_num_used = self.cond_num
        cond_cat_used = self.cond_cat_soft
        cond_y_used = self.cond_y_soft

        if cond_num_used is not None and x_num_t.shape[0] > cond_num_used.shape[0]:
            K_copies = x_num_t.shape[0] // cond_num_used.shape[0]
            cond_num_used = cond_num_used.repeat_interleave(K_copies, dim=0)
            if cond_cat_used is not None: cond_cat_used = cond_cat_used.repeat_interleave(K_copies, dim=0)
            if cond_y_used is not None: cond_y_used = cond_y_used.repeat_interleave(K_copies, dim=0)

        x_num_comb = torch.cat([x_num_t, cond_num_used], dim=1) if cond_num_used is not None else x_num_t
        cat_parts = [x_cat_t_soft]
        if cond_cat_used is not None and cond_cat_used.shape[1] > 0: cat_parts.append(cond_cat_used)
        if cond_y_used is not None: cat_parts.append(cond_y_used)
            
        x_cat_comb = torch.cat(cat_parts, dim=1) if len(cat_parts) > 1 else x_cat_t_soft
        out_num_comb, out_cat_comb = self.base_model(x_num_comb, x_cat_comb, t)
        return out_num_comb[:, :self.num_num], out_cat_comb[:, :self.sum_cat_t]

def build_diffusion_model(cfg, dataset, device):
    num_dim = dataset.num_numerical
    cat_dims = dataset.cat_cardinalities 
    y_classes = max(2, dataset.num_classes_target)
    cat_dims_w_mask = [c + 1 for c in cat_dims]
    d_numerical_aug = num_dim * 2
    categories_aug = cat_dims_w_mask + cat_dims_w_mask + [y_classes]
    
    unimod = UniModMLP(
        d_numerical=d_numerical_aug, categories=categories_aug,
        num_layers=cfg.get('model', {}).get('num_layers', 4), d_token=cfg.get('model', {}).get('d_token', 64),
        n_head=cfg.get('model', {}).get('n_head', 1), factor=cfg.get('model', {}).get('factor', 4),
        bias=True, dim_t=cfg.get('model', {}).get('dim_t', 256), use_mlp=True
    )
    sum_cat_t = sum(cat_dims_w_mask)
    cond_denoiser = ConditionalDenoiser(unimod, num_dim, sum_cat_t)
    denoise_fn = Model(denoise_fn=cond_denoiser, sigma_data=1.0, precond=True, net_conditioning="sigma")
    T = cfg.get('diffusion', {}).get('T', 100)
    
    tabdiff = UnifiedCtimeDiffusion(
        num_classes=np.array(cat_dims), num_numerical_features=num_dim,
        denoise_fn=denoise_fn, y_only_model=None, num_timesteps=T,
        scheduler='power_mean_per_column', cat_scheduler='log_linear_per_column',
        noise_dist='uniform_t', edm_params={'sigma_data': 1.0},
        sampler_params={'stochastic_sampler': True, 'second_order_correction': True}, device=device
    ).to(device)
    tabdiff.set_condition = cond_denoiser.set_condition
    return tabdiff

def tabdiff_to_flat_tensor(z_norm, z_cat, cat_cardinalities):
    cat_parts = []
    if z_cat is not None and z_cat.shape[1] > 0:
        for i, card in enumerate(cat_cardinalities):
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


@torch.no_grad()
def sample_counterfactuals_tabdiff(diffusion_model, x_num_orig, x_cat_orig, noise_level=0.6, temperature=1.0):
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

    for i in reversed(range(0, start_step + 1)):
        z_norm, z_cat, _ = diffusion_model.edm_update(
            z_norm, z_cat, i, t[i], t[i-1] if i > 0 else None, t[i],
            sigma_num_cur[i], sigma_num_next[i], sigma_num_cur[i], 
            sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_cur[i],
        )
    return z_norm, z_cat


def greedy_dpp(L, k):

    item_size = L.shape[0]
    if item_size <= k:
        return torch.arange(item_size, device=L.device).tolist()
        
    cis = torch.zeros((k, item_size), device=L.device)
    di2s = torch.diag(L).clone()
    selected_items = []
    selected_mask = torch.zeros(item_size, dtype=torch.bool, device=L.device)
    
    # Wybieramy pierwszy punkt o najwyższej wariancji (najwyższa jakość)
    j = torch.argmax(di2s).item()
    selected_items.append(j)
    selected_mask[j] = True
    
    while len(selected_items) < k:
        k_curr = len(selected_items) - 1
        j_idx = selected_items[-1]
        
        if k_curr == 0:
            e = L[j_idx, ~selected_mask] / torch.sqrt(di2s[j_idx] + 1e-8)
        else:
            c_j = cis[:k_curr, j_idx]
            c_avail = cis[:k_curr, ~selected_mask]
            inner_prod = torch.matmul(c_j, c_avail)
            e = (L[j_idx, ~selected_mask] - inner_prod) / torch.sqrt(di2s[j_idx] + 1e-8)
        
        cis[k_curr, ~selected_mask] = e
        di2s[~selected_mask] -= e ** 2

        avail_idx = torch.where(~selected_mask)[0]
        if len(avail_idx) == 0:
            break
            
        best_avail = torch.argmax(di2s[~selected_mask])
        next_item = avail_idx[best_avail].item()
        
        selected_items.append(next_item)
        selected_mask[next_item] = True
        
    return selected_items

def load_config(path: str) -> dict:
    with open(path, 'r') as f: return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dpp_infer.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TABDIFF + DPP FILTERING | DEVICE: {device} ===")

    data_dir = cfg['dataset'].get('data_dir', 'data/adult')
    with open(os.path.join(data_dir, "config.json"), 'r') as f: data_config = json.load(f)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, spec = get_generic_data(data_dir, data_config)

    train_dataset = TabularCounterfactualDataset(X=X_train_raw, y=y_train, spec=spec, k=cfg['dataset']['k_neighbors'], device=device, build_neighbors=True)
    val_dataset = TabularCounterfactualDataset(X=X_val_raw, y=y_val, spec=spec, k=cfg['dataset']['k_neighbors'], device=device, scaler=train_dataset.scaler, ordinal_encoder=train_dataset.ordinal_encoder, build_neighbors=False) if X_val_raw is not None else None

    ckpt_path = cfg['diffusion']['checkpoint_path']
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    orig_cfg = checkpoint.get("config", cfg)
    if "model" in orig_cfg: cfg["model"] = orig_cfg["model"]
    if "classifier" in orig_cfg: cfg["classifier"] = orig_cfg["classifier"]

    print("\n[Loading Baseline NN Classifier]")
    cfg['classifier']['epochs'] = 0 
    clf_model_nn = train_classifier(train_dataset, cfg['classifier'], device, val_dataset)
    if "clf_state" in checkpoint and checkpoint["clf_state"] is not None:
        clf_model_nn.load_state_dict(checkpoint["clf_state"])
    clf_model_nn.eval()
    clf_flat = ClfFlatWrapper(clf_model_nn, train_dataset.num_numerical)
    train_dataset.relabel(clf_model_nn, device=device)

    print("[Loading TabDiff Diffusion Model]")
    trained_diffusion = build_diffusion_model(cfg, train_dataset, device)
    trained_diffusion.load_state_dict(checkpoint.get("model_state", checkpoint))
    trained_diffusion.eval()

    # Parametry generacji i filtracji
    NUM_TEST = cfg['dataset'].get('n_test_samples', 500)
    N_GENERATED = cfg['dataset'].get('n_cf_generated', 50) 
    N_SELECTED = cfg['dataset'].get('n_cf_selected', 10)
    NOISE_LEVEL = cfg['diffusion'].get('noise_level', 0.6)
    
    indices = np.random.choice(len(X_test_raw), min(len(X_test_raw), NUM_TEST), replace=False)
    test_dataset_helper = TabularCounterfactualDataset(X=X_test_raw[indices], y=y_test[indices], spec=spec, device=device, scaler=train_dataset.scaler, ordinal_encoder=train_dataset.ordinal_encoder, build_neighbors=False)
    
    x_num_orig = test_dataset_helper.X_num
    x_cat_orig = test_dataset_helper.X_cat
    x_orig_flat = test_dataset_helper.X_model 
    y_test_tensor = test_dataset_helper.y
    y_target = torch.clamp((y_test_tensor + 1) % 2, 0, 1)

    print(f"\n--- Over-generating {N_GENERATED} CFs per sample ---")
    x_num_expanded = x_num_orig.repeat_interleave(N_GENERATED, dim=0)
    x_cat_expanded = x_cat_orig.repeat_interleave(N_GENERATED, dim=0)
    y_input_expanded = y_target.repeat_interleave(N_GENERATED, dim=0)

    with torch.no_grad():
        cond_cat_parts = []
        if x_cat_expanded.shape[1] > 0:
            for i, card in enumerate(train_dataset.cat_cardinalities):
                cond_cat_parts.append(F.one_hot(x_cat_expanded[:, i].long(), num_classes=card+1).float())
            cond_cat_soft = torch.cat(cond_cat_parts, dim=-1)
        else:
            cond_cat_soft = torch.empty(x_num_expanded.shape[0], 0, device=device)
        cond_y_soft = F.one_hot(y_input_expanded.long(), num_classes=max(2, train_dataset.num_classes_target)).float()
        trained_diffusion.set_condition(x_num_expanded, cond_cat_soft, cond_y_soft)

        # Generacja masowa (Vanilla)
        z_n_vanilla, z_c_vanilla = sample_counterfactuals_tabdiff(trained_diffusion, x_num_expanded, x_cat_expanded, noise_level=NOISE_LEVEL)
        all_cfs_flat = tabdiff_to_flat_tensor(z_n_vanilla, z_c_vanilla, train_dataset.cat_cardinalities)
        
        # Pytamy klasyfikator, kto przeszedł granicę decyzyjną
        logits = clf_flat(all_cfs_flat)
        probs = F.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

    print(f"\n--- Filtering, DPP & Random Selection (Target: {N_SELECTED} per sample) ---")
    
    final_cfs_dpp_list = []
    final_cfs_random_list = []
    total_valid_found = 0
    
    for i in range(len(x_orig_flat)):
        start_idx = i * N_GENERATED
        end_idx = start_idx + N_GENERATED
        
        cfs_i = all_cfs_flat[start_idx:end_idx]
        preds_i = pred_classes[start_idx:end_idx]
        target_i = y_target[i].item()
        valid_mask = (preds_i == target_i)
        valid_cfs = cfs_i[valid_mask]
        num_valid = valid_cfs.shape[0]
        
        selected_cfs_dpp = []
        selected_cfs_random = []
        
        if num_valid >= N_SELECTED:
            total_valid_found += N_SELECTED

            orig_flat_for_sample = x_orig_flat[i:i+1]
            dists = torch.norm(valid_cfs - orig_flat_for_sample, dim=1)
            sigma_q = dists.mean() + 1e-5
            q = torch.exp(-dists / sigma_q)
            diffs = valid_cfs.unsqueeze(1) - valid_cfs.unsqueeze(0)
            pairwise_dists = torch.norm(diffs, dim=2)
            sigma_s = pairwise_dists.mean() + 1e-5
            S = torch.exp(-pairwise_dists / sigma_s)

            L = torch.outer(q, q) * S
            idx_dpp = greedy_dpp(L, N_SELECTED)
            selected_cfs_dpp = valid_cfs[idx_dpp]
            idx_random = torch.randperm(num_valid)[:N_SELECTED]
            selected_cfs_random = valid_cfs[idx_random]
            
        else:
            total_valid_found += num_valid
            shortage = N_SELECTED - num_valid
            invalid_cfs = cfs_i[~valid_mask]
            
            fallback_cfs = [valid_cfs]
            if invalid_cfs.shape[0] > 0:
                invalid_probs = probs[start_idx:end_idx][~valid_mask]
                target_probs = invalid_probs[:, int(target_i)]
                _, top_invalid_idx = torch.sort(target_probs, descending=True)
                fallback_cfs.append(invalid_cfs[top_invalid_idx[:shortage]])
            
            fallback_tensor = torch.cat(fallback_cfs, dim=0)
            selected_cfs_dpp = fallback_tensor
            selected_cfs_random = fallback_tensor.clone()
            
        final_cfs_dpp_list.append(selected_cfs_dpp)
        final_cfs_random_list.append(selected_cfs_random)

    final_cfs_dpp_tensor = torch.cat(final_cfs_dpp_list, dim=0)
    final_cfs_random_tensor = torch.cat(final_cfs_random_list, dim=0)
    
    validity_ratio = total_valid_found / (len(x_orig_flat) * N_SELECTED)
    print(f"[Info] Percentage of purely VALID counterfactuals available for selection: {validity_ratio * 100:.2f}%")

    print("\n--- Evaluating Both Selections ---")
    evaluator = MetricsEvaluator(train_dataset)
    X_train_np_inv = train_dataset.inverse_transform(train_dataset.X_model)
    
    x_flat_eval = x_orig_flat.repeat_interleave(N_SELECTED, dim=0)
    y_input_eval = y_target.repeat_interleave(N_SELECTED, dim=0)
    group_ids_eval = np.arange(len(x_orig_flat)).repeat(N_SELECTED)

    metrics_dpp = evaluator.evaluate(
        x_orig_tensor=x_flat_eval, x_cf_tensor=final_cfs_dpp_tensor, y_target_tensor=y_input_eval, 
        cf_group_ids=group_ids_eval, X_train_np=X_train_np_inv, clf_model=clf_flat 
    )

    metrics_random = evaluator.evaluate(
        x_orig_tensor=x_flat_eval, x_cf_tensor=final_cfs_random_tensor, y_target_tensor=y_input_eval, 
        cf_group_ids=group_ids_eval, X_train_np=X_train_np_inv, clf_model=clf_flat 
    )

    df_results = pd.DataFrame([
        {"Experiment": "TabDiff_DPP_Selection", **metrics_dpp},
        {"Experiment": "TabDiff_Random_Selection", **metrics_random}
    ])
    

    cols_to_print = ["Experiment", "validity", "prox_cont", "spars_cat", "diversity", "lof_inliers"]
    existing_cols = [c for c in cols_to_print if c in df_results.columns]
    
    print(df_results[existing_cols].to_string(index=False))

    results_file = cfg.get("results_path", "dpp_vs_random_results.csv")
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    df_results.to_csv(results_file, index=False)
    print(f"\n[SUCCESS] Results saved to: {results_file}")

if __name__ == "__main__":
    main()