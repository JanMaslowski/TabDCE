import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.linear_model import LogisticRegression
import time

from tabdce.loops.train_classifier import train_classifier
from tabdce.dataset.dataset import TabularCounterfactualDataset, get_generic_data
from tabdce.utils.advanced_metrics import MetricsEvaluator, generate_knn_counterfactuals

from tabdce.model.modules import UniModMLP, Model
from tabdce.model.diffusion import UnifiedCtimeDiffusion



def greedy_dpp(L, k):
    item_size = L.shape[0]
    if item_size <= k:
        return torch.arange(item_size, device=L.device).tolist()
        
    cis = torch.zeros((k, item_size), device=L.device)
    di2s = torch.diag(L).clone()
    selected_items = []
    selected_mask = torch.zeros(item_size, dtype=torch.bool, device=L.device)
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


def check_actionability_constraints(x_num_orig, x_cat_orig, x_num_cf, x_cat_cf, imm_num_idx, imm_cat_idx, inc_num_idx, inc_cat_idx, ranges_tensor, epsilon=0.05):
    b = x_num_orig.shape[0]
    device = x_num_orig.device
    
    violation_counts = torch.zeros(b, dtype=torch.int32, device=device)
    thresholds = epsilon * ranges_tensor.view(1, -1) if ranges_tensor.numel() > 0 else torch.empty(0, device=device)

    if imm_num_idx:
        diff = torch.abs(x_num_cf[:, imm_num_idx] - x_num_orig[:, imm_num_idx])
        violation_counts += (diff > thresholds[:, imm_num_idx]).int().sum(dim=1)

    if inc_num_idx:
        diff = x_num_orig[:, inc_num_idx] - x_num_cf[:, inc_num_idx]
        violation_counts += (diff > thresholds[:, inc_num_idx]).int().sum(dim=1)

    if imm_cat_idx and x_cat_orig.shape[1] > 0:
        cf_cat_rounded = torch.round(x_cat_cf[:, imm_cat_idx])
        orig_cat_rounded = torch.round(x_cat_orig[:, imm_cat_idx])
        violation_counts += (cf_cat_rounded != orig_cat_rounded).int().sum(dim=1)
        
    if inc_cat_idx and x_cat_orig.shape[1] > 0:
        cf_cat_rounded = torch.round(x_cat_cf[:, inc_cat_idx])
        orig_cat_rounded = torch.round(x_cat_orig[:, inc_cat_idx])
        violation_counts += (cf_cat_rounded < orig_cat_rounded).int().sum(dim=1)

    mask_actionable = (violation_counts == 0)
    actionability_rate = mask_actionable.float().mean().item()

    unique_vals, counts = torch.unique(violation_counts, return_counts=True)
    distribution_dict = {f"Violated_{val}_Rules_%": (count / float(b)) * 100.0 for val, count in zip(unique_vals.tolist(), counts.tolist())}

    return actionability_rate, mask_actionable, distribution_dict

class ConditionalDenoiser(nn.Module):
    def __init__(self, base_model: nn.Module, num_num: int, sum_cat_t: int, drop_prob: float = 0.25):
        super().__init__()
        self.base_model = base_model
        self.num_num = num_num
        self.sum_cat_t = sum_cat_t
        self.drop_prob = drop_prob
        self.cond_num = None
        self.cond_cat_soft = None
        self.cond_y_soft = None

    def set_condition(self, cond_num, cond_cat_soft, cond_y_soft):
        self.cond_num = cond_num
        self.cond_cat_soft = cond_cat_soft
        self.cond_y_soft = cond_y_soft

    def forward(self, x_num_t, x_cat_t_soft, t, sigma=None):
        apply_drop = self.training and (torch.rand(1).item() < self.drop_prob)
        cond_num_used = torch.zeros_like(self.cond_num) if apply_drop and self.cond_num is not None else self.cond_num
        cond_cat_used = torch.zeros_like(self.cond_cat_soft) if apply_drop and self.cond_cat_soft is not None else self.cond_cat_soft
        cond_y_used = torch.zeros_like(self.cond_y_soft) if apply_drop and self.cond_y_soft is not None else self.cond_y_soft

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

class ClfLRBase(nn.Module):
    def __init__(self, lr_sklearn, device):
        super().__init__()
        self.coef = torch.tensor(lr_sklearn.coef_, dtype=torch.float32, device=device)
        self.intercept = torch.tensor(lr_sklearn.intercept_, dtype=torch.float32, device=device)
        
    def forward(self, x_num, x_cat):
        x_flat = torch.cat([x_num, x_cat], dim=1)
        logits_class_1 = F.linear(x_flat, self.coef, self.intercept)
        logits_class_0 = -logits_class_1 
        return torch.cat([logits_class_0, logits_class_1], dim=1)


@torch.no_grad()
def sample_counterfactuals_svdd(
    diffusion_model, clf_model, x_num_orig, x_cat_orig, y_target, cat_cardinalities,
    ranges_tensor, noise_level=0.8, K=5, guidance_scale=20.0, dist_scale=0.1, cat_scale=1.0, temperature=1.0,
    imm_num_idx=None, imm_cat_idx=None, inc_num_idx=None, inc_cat_idx=None, constraint_weight=100.0, epsilon=0.05
):
    device = diffusion_model.device
    b = x_num_orig.shape[0]
    
    imm_num_idx = imm_num_idx or []
    imm_cat_idx = imm_cat_idx or []
    inc_num_idx = inc_num_idx or []
    inc_cat_idx = inc_cat_idx or []
    
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
        z_norm_k = z_norm.repeat_interleave(K, dim=0)
        z_cat_k = z_cat.repeat_interleave(K, dim=0) if z_cat.shape[1] > 0 else z_cat
        x_num_orig_k = x_num_orig.repeat_interleave(K, dim=0) 
        x_cat_orig_k = x_cat_orig.repeat_interleave(K, dim=0) 
        y_target_k = y_target.repeat_interleave(K, dim=0)
        
        z_norm_next, z_cat_next, _ = diffusion_model.edm_update(
            z_norm_k, z_cat_k, i, t[i], t[i-1] if i > 0 else None, t[i],
            sigma_num_cur[i], sigma_num_next[i], sigma_num_cur[i], 
            sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_cur[i],
        )

        t_curr = t[i].squeeze().repeat(b * K).to(device)
        sigma_curr = sigma_num_cur[i].unsqueeze(0).repeat(b * K, 1).to(device)

        z_cat_next_oh = diffusion_model.to_one_hot(z_cat_next).to(z_norm_next.dtype) if z_cat_next.shape[1] > 0 else z_cat_next
        x0_hat_num, x0_hat_cat = diffusion_model._denoise_fn(z_norm_next.float(), z_cat_next_oh, t_curr, sigma=sigma_curr)
        
        if x0_hat_cat.shape[1] > 0:
            cat_indices = []
            start_idx = 0
            for card in cat_cardinalities:
                idx = x0_hat_cat[:, start_idx : start_idx + card].argmax(dim=-1)
                cat_indices.append(idx.unsqueeze(1))
                start_idx += card + 1
            cat_candidates = torch.cat(cat_indices, dim=1)
            cat_changes_count = (cat_candidates != x_cat_orig_k).float().sum(dim=1)
        else:
            cat_candidates = torch.empty(b * K, 0, device=device)
            cat_changes_count = torch.zeros(b * K, device=device)
            
        dist_num_sq = torch.sum((x0_hat_num - x_num_orig_k) ** 2, dim=1)
        x0_flat = torch.cat([x0_hat_num, cat_candidates.float()], dim=1) if cat_candidates.shape[1] > 0 else x0_hat_num
            
        clf_logits = clf_model(x0_flat)
        validity_logits = clf_logits.gather(1, y_target_k.unsqueeze(1).long()).squeeze(1)
        validity_reward = torch.clamp(validity_logits, max=2.0)
        
        constraint_penalty = torch.zeros(b * K, device=device)

        if imm_num_idx:
            diff_imm = torch.abs(x0_hat_num[:, imm_num_idx] - x_num_orig_k[:, imm_num_idx])
            constraint_penalty += torch.sum(diff_imm, dim=1)
        if inc_num_idx:
            diff_inc = F.relu(x_num_orig_k[:, inc_num_idx] - x0_hat_num[:, inc_num_idx])
            constraint_penalty += torch.sum(diff_inc, dim=1)
        if imm_cat_idx and z_cat.shape[1] > 0:
            diff_cat = (cat_candidates[:, imm_cat_idx] != x_cat_orig_k[:, imm_cat_idx]).float()
            constraint_penalty += torch.sum(diff_cat, dim=1)
        if inc_cat_idx and z_cat.shape[1] > 0:
            diff_inc_cat = F.relu(x_cat_orig_k[:, inc_cat_idx] - cat_candidates[:, inc_cat_idx])
            constraint_penalty += torch.sum(diff_inc_cat, dim=1)
            
        total_reward = validity_reward - (dist_scale * dist_num_sq) - (cat_scale * cat_changes_count) - (constraint_weight * constraint_penalty)

        current_t = t[i].item()
        dynamic_guidance = guidance_scale * (current_t + 0.2)

        scaled_reward = total_reward.view(b, K) * dynamic_guidance
        scaled_reward = torch.nan_to_num(scaled_reward, nan=-1e4, posinf=1e4, neginf=-1e4)
        scaled_reward = torch.clamp(scaled_reward, min=-100.0, max=100.0)
        
        if i == 0:
            chosen_idx = torch.argmax(scaled_reward, dim=1)
        else:
            weights = F.softmax(scaled_reward, dim=1) + 1e-9
            weights = weights / weights.sum(dim=1, keepdim=True) 
            chosen_idx = torch.multinomial(weights, 1).squeeze(1)
            
        batch_offsets = torch.arange(b, device=device) * K
        global_chosen_idx = batch_offsets + chosen_idx
        
        z_norm = z_norm_next[global_chosen_idx]
        if z_cat.shape[1] > 0:
            z_cat = z_cat_next[global_chosen_idx]

        if imm_num_idx:
            z_norm[:, imm_num_idx] = x_num_orig[:, imm_num_idx]
        if inc_num_idx:
            mask_decreased = z_norm[:, inc_num_idx] < x_num_orig[:, inc_num_idx]
            z_norm[:, inc_num_idx] = torch.where(mask_decreased, x_num_orig[:, inc_num_idx], z_norm[:, inc_num_idx])
        if imm_cat_idx and z_cat.shape[1] > 0:
            z_cat[:, imm_cat_idx] = x_cat_orig[:, imm_cat_idx]
        if inc_cat_idx and z_cat.shape[1] > 0:
            mask_cat_decreased = z_cat[:, inc_cat_idx] < x_cat_orig[:, inc_cat_idx]
            z_cat[:, inc_cat_idx] = torch.where(mask_cat_decreased, x_cat_orig[:, inc_cat_idx], z_cat[:, inc_cat_idx])

    return z_norm, z_cat


def load_config(path: str) -> dict:
    with open(path, 'r') as f: return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SVDD POST-SELECTION EXPERIMENT (DPP vs Random) | DEVICE: {device} ===")

    data_dir = cfg['dataset'].get('data_dir', 'data/adult')
    with open(os.path.join(data_dir, "config.json"), 'r') as f: data_config = json.load(f)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, spec = get_generic_data(data_dir, data_config)

    train_dataset = TabularCounterfactualDataset(X=X_train_raw, y=y_train, spec=spec, k=cfg['dataset']['k_neighbors'], device=device, build_neighbors=False)
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
    clf_flat_nn = ClfFlatWrapper(clf_model_nn, train_dataset.num_numerical)

    if 'new_mlp' in cfg and cfg['new_mlp'].get('train', False):
        new_mlp_params = cfg['new_mlp']
        clf_model_sec = train_classifier(train_dataset, new_mlp_params, device, val_dataset)
        clf_model_sec.eval()
        clf_flat_sec = ClfFlatWrapper(clf_model_sec, train_dataset.num_numerical)
        sec_guidance_key = "new_mlp"
    else:
        lr_model_sklearn = LogisticRegression(max_iter=2000)
        lr_model_sklearn.fit(train_dataset.X_model.cpu().numpy(), train_dataset.y.cpu().numpy())
        clf_model_sec = ClfLRBase(lr_model_sklearn, device)
        clf_flat_sec = ClfFlatWrapper(clf_model_sec, train_dataset.num_numerical)
        sec_guidance_key = "lr"

    print("\n[Loading TabDiff Diffusion Model]")
    trained_diffusion = build_diffusion_model(cfg, train_dataset, device)
    trained_diffusion.load_state_dict(checkpoint.get("model_state", checkpoint))
    trained_diffusion.eval()

    NUM_TEST = cfg['dataset'].get('n_test_samples', 500)
    N_CF_POOL = 100   
    N_CF_TARGET = 10  
    
    indices = np.random.choice(len(X_test_raw), min(len(X_test_raw), NUM_TEST), replace=False)
    NUM_TEST = len(indices)
    
    test_dataset_helper = TabularCounterfactualDataset(X=X_test_raw[indices], y=y_test[indices], spec=spec, device=device, scaler=train_dataset.scaler, ordinal_encoder=train_dataset.ordinal_encoder, build_neighbors=False)
    
    x_num_orig = test_dataset_helper.X_num
    x_cat_orig = test_dataset_helper.X_cat
    x_orig_flat = test_dataset_helper.X_model 
    y_test_tensor = test_dataset_helper.y
    y_target = torch.clamp((y_test_tensor + 1) % 2, 0, 1)

    evaluator = MetricsEvaluator(train_dataset)
    X_train_np_inv = train_dataset.inverse_transform(train_dataset.X_model)

    if train_dataset.num_numerical > 0:
        ranges_tensor = train_dataset.X_num.max(dim=0)[0] - train_dataset.X_num.min(dim=0)[0]
    else:
        ranges_tensor = torch.empty(0, device=device)

    all_results = []
    
    experiments = cfg.get("svdd_experiments", [])
    print(f"\n=== Starting Post-Selection Experiments ({len(experiments)} configurations) ===")
    
    for exp_idx, exp in enumerate(experiments):
        exp_name = exp.get("name", f"Exp_{exp_idx}")
        clf_type = exp.get("clf_guidance", "nn").lower()
        active_clf_flat = clf_flat_sec if clf_type == sec_guidance_key else clf_flat_nn
        active_clf_base = clf_model_sec if clf_type == sec_guidance_key else clf_model_nn
        
        train_dataset.relabel(active_clf_base, device=device)

        K_param = exp.get("K", 5)
        exp_imm_num = exp.get("immutable_num_idx", [])
        exp_imm_cat = exp.get("immutable_cat_idx", [])
        exp_inc_num = exp.get("increasing_num_idx", [])
        exp_inc_cat = exp.get("increasing_cat_idx", [])
        
        print(f"\n---> Running: {exp_name} | Generating {N_CF_POOL} candidates per sample via SVDD")

        # Przygotowanie puli wejściowej [B * 100]
        x_num_pool = x_num_orig.repeat_interleave(N_CF_POOL, dim=0)
        x_cat_pool = x_cat_orig.repeat_interleave(N_CF_POOL, dim=0)
        x_flat_pool = x_orig_flat.repeat_interleave(N_CF_POOL, dim=0)
        y_input_pool = y_target.repeat_interleave(N_CF_POOL, dim=0)
        
        with torch.no_grad():
            cond_cat_parts = []
            if x_cat_pool.shape[1] > 0:
                for i, card in enumerate(train_dataset.cat_cardinalities):
                    cond_cat_parts.append(F.one_hot(x_cat_pool[:, i].long(), num_classes=card+1).float())
                cond_cat_soft_pool = torch.cat(cond_cat_parts, dim=-1)
            else:
                cond_cat_soft_pool = torch.empty(x_num_pool.shape[0], 0, device=device)
            cond_y_soft_pool = F.one_hot(y_input_pool.long(), num_classes=max(2, train_dataset.num_classes_target)).float()
            trained_diffusion.set_condition(x_num_pool, cond_cat_soft_pool, cond_y_soft_pool)


        with torch.no_grad():
            z_n_svdd, z_c_svdd = sample_counterfactuals_svdd(
                diffusion_model=trained_diffusion, clf_model=active_clf_flat,  
                x_num_orig=x_num_pool, x_cat_orig=x_cat_pool, y_target=y_input_pool,
                cat_cardinalities=train_dataset.cat_cardinalities,
                ranges_tensor=ranges_tensor, 
                noise_level=exp.get("noise_level", 0.6), K=K_param, 
                guidance_scale=exp.get("guidance_scale", 20.0), dist_scale=exp.get("dist_scale", 0.1), cat_scale=exp.get("cat_scale", 1.0),
                imm_num_idx=exp_imm_num, imm_cat_idx=exp_imm_cat, inc_num_idx=exp_inc_num, inc_cat_idx=exp_inc_cat, 
                constraint_weight=exp.get("constraint_weight", 200.0), epsilon=0.05
            )
            cfs_svdd_flat = tabdiff_to_flat_tensor(z_n_svdd, z_c_svdd, train_dataset.cat_cardinalities)
        _, mask_act, _ = check_actionability_constraints(
            x_num_pool, x_cat_pool, z_n_svdd, z_c_svdd, 
            exp_imm_num, exp_imm_cat, exp_inc_num, exp_inc_cat, ranges_tensor
        )
        
        with torch.no_grad():
            clf_logits = active_clf_flat(cfs_svdd_flat)
            preds = clf_logits.argmax(dim=1) if clf_logits.shape[-1] > 1 else (torch.sigmoid(clf_logits) > 0.5).long()
            mask_valid = (preds == y_input_pool).squeeze()
            
        joint_mask = mask_act & mask_valid
        

        joint_mask_grouped = joint_mask.view(NUM_TEST, N_CF_POOL)
        valid_counts_per_sample = joint_mask_grouped.sum(dim=1)
        eligible_samples_mask = valid_counts_per_sample >= N_CF_TARGET
        coverage_percent = eligible_samples_mask.float().mean().item() * 100.0
        print(f"     [Coverage] {coverage_percent:.2f}% of test samples produced >= {N_CF_TARGET} valid/actionable CFs.")

        if coverage_percent == 0:
            print("     [Warning] No samples met the threshold. Skipping metrics evaluation for this config.")
            all_results.append({"Experiment": exp_name, "Coverage_%": 0.0})
            continue

        dpp_cfs, dpp_origs, dpp_tgts, dpp_groups = [], [], [], []
        rnd_cfs, rnd_origs, rnd_tgts, rnd_groups = [], [], [], []
        
        for b_idx in range(NUM_TEST):
            if not eligible_samples_mask[b_idx]: continue
                
            start_idx = b_idx * N_CF_POOL
            end_idx = start_idx + N_CF_POOL

            local_valid_indices = torch.where(joint_mask_grouped[b_idx])[0]
            global_valid_indices = start_idx + local_valid_indices
            
            valid_cfs_for_sample = cfs_svdd_flat[global_valid_indices]
            orig_flat_for_sample = x_orig_flat[b_idx:b_idx+1]
            tgt_for_sample = y_target[b_idx:b_idx+1]

            rnd_local_idx = torch.randperm(len(local_valid_indices))[:N_CF_TARGET]
            rnd_cfs.append(valid_cfs_for_sample[rnd_local_idx])
            rnd_origs.append(orig_flat_for_sample.repeat(N_CF_TARGET, 1))
            rnd_tgts.append(tgt_for_sample.repeat(N_CF_TARGET))
            rnd_groups.extend([b_idx] * N_CF_TARGET)
            
            dists = torch.norm(valid_cfs_for_sample - orig_flat_for_sample, dim=1)
            sigma_q = dists.mean() + 1e-5
            q = torch.exp(-dists / sigma_q)
            diffs = valid_cfs_for_sample.unsqueeze(1) - valid_cfs_for_sample.unsqueeze(0)
            pairwise_dists = torch.norm(diffs, dim=2)
            sigma_s = pairwise_dists.mean() + 1e-5
            S = torch.exp(-pairwise_dists / sigma_s)
            L = torch.outer(q, q) * S
            
            dpp_local_idx = greedy_dpp(L, N_CF_TARGET)
            dpp_cfs.append(valid_cfs_for_sample[dpp_local_idx])
            dpp_origs.append(orig_flat_for_sample.repeat(N_CF_TARGET, 1))
            dpp_tgts.append(tgt_for_sample.repeat(N_CF_TARGET))
            dpp_groups.extend([b_idx] * N_CF_TARGET)

        metrics_rnd = evaluator.evaluate(
            x_orig_tensor=torch.cat(rnd_origs), x_cf_tensor=torch.cat(rnd_cfs), 
            y_target_tensor=torch.cat(rnd_tgts), cf_group_ids=np.array(rnd_groups), 
            X_train_np=X_train_np_inv, clf_model=active_clf_flat
        )
        all_results.append({
            "Experiment": f"{exp_name}_1_SVDD_Random", "Coverage_%": coverage_percent, **metrics_rnd
        })
        metrics_dpp = evaluator.evaluate(
            x_orig_tensor=torch.cat(dpp_origs), x_cf_tensor=torch.cat(dpp_cfs), 
            y_target_tensor=torch.cat(dpp_tgts), cf_group_ids=np.array(dpp_groups), 
            X_train_np=X_train_np_inv, clf_model=active_clf_flat
        )
        all_results.append({
            "Experiment": f"{exp_name}_2_SVDD_DPP", "Coverage_%": coverage_percent, **metrics_dpp
        })

    df_results = pd.DataFrame(all_results)
    print("\n=== FINAL RESULTS OVERVIEW ===")
    cols_to_print = ["Experiment", "Coverage_%", "Proximity_Num", "Sparsity_Cat", "diversity"]
    existing_cols = [c for c in cols_to_print if c in df_results.columns]
    print(df_results[existing_cols].to_string(index=False))

    results_file = cfg.get("results_path", "svdd_postselection_results.csv")
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    df_results.to_csv(results_file, index=False)
    print(f"\n[SUCCESS] Post-selection experiment results saved to: {results_file}")

if __name__ == "__main__":
    main()