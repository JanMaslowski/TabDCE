from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabdce.utils.utils import extract, DiffusionSchedule

class MixedTabularDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        num_numerical: int,
        num_classes: List[int],  # Lista kardynalności [K1, K2...]
        T: int = 1000,
        schedule: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.num_numerical = num_numerical
        self.num_classes = num_classes
        self.num_classes_tensor = torch.tensor(num_classes, device=device)
        self.total_cat_dim = sum(num_classes)
        self.device = device

        sched = DiffusionSchedule.from_name(schedule, T, device)
        self.register_buffer("betas", sched.betas)
        alphas = 1.0 - sched.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)
        
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        
        self.register_buffer("posterior_variance", sched.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer("posterior_mean_coef1", sched.betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar))

    def q_sample_gauss(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion for numerical variables."""
        coef1 = extract(self.sqrt_alphas_bar, t, x0.shape)
        coef2 = extract(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        return coef1 * x0 + coef2 * noise

    def p_mean_variance_gauss(self, x_t: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and variance of p(x_{t-1} | x_t) given predicted noise (eps).
        Uses the standard DDPM formulation re-parameterized to predict eps.
        """

        sqrt_recip_alphas_bar = extract(1.0 / self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_recip_m1_alphas_bar = extract(torch.sqrt(1.0 / self.alphas_bar - 1.0), t, x_t.shape)
        pred_x0 = sqrt_recip_alphas_bar * x_t - sqrt_recip_m1_alphas_bar * pred_eps
        pred_x0 = pred_x0.clamp(-5.0, 5.0) 
        post_mean_coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        post_mean_coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        model_mean = post_mean_coef1 * pred_x0 + post_mean_coef2 * x_t
        model_log_var = extract(self.posterior_variance.clamp(min=1e-20).log(), t, x_t.shape)
        return model_mean, model_log_var

    def q_sample_cat(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Forward diffusion for categorical (Multinomial).
            Uniform noise transition.
            """
            B = log_x0.shape[0]
            alphas_bar_t = extract(self.alphas_bar, t, (B, 1))
            
            out_parts = []
            start = 0
            for K in self.num_classes:
                sl = slice(start, start+K)
                log_probs_x0 = log_x0[:, sl]
                log_alpha = alphas_bar_t.log()
                log_1_m_alpha = (1. - alphas_bar_t).log()
                
                log_inv_K = -torch.log(torch.tensor(K, device=log_x0.device))
                
                term1 = log_alpha + log_probs_x0
                term2 = log_1_m_alpha + log_inv_K
                
                log_probs_t = torch.logaddexp(term1, term2)
                uniform = torch.rand_like(log_probs_t).clamp(min=1e-30)
                gumbel = -torch.log(-torch.log(uniform))
                sample_t = F.one_hot((log_probs_t + gumbel).argmax(dim=1), num_classes=K).float()
                
                out_parts.append((sample_t + 1e-30).log())
                start += K
                
            return torch.cat(out_parts, dim=1)

    def p_sample_cat(self, log_x_t: torch.Tensor, t: torch.Tensor, log_pred_x0: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample x_{t-1} given x_t and predicted x_0 (logits).
        Calculates posterior q(x_{t-1} | x_t, x_0).
        """
        out_parts = []
        start = 0
        t_idx = t[0] 
        
        for K in self.num_classes:
            sl = slice(start, start+K)
            log_x0_rec = log_pred_x0[:, sl]  
            
            log_x0_rec = F.log_softmax(log_x0_rec, dim=1)
            out_parts.append(log_x0_rec) 
                
            start += K
            
        full_log_probs = torch.cat(out_parts, dim=1)
        return self._sample_cat_from_logits(full_log_probs, temperature=temperature)

    def _sample_cat_from_logits(self, logits_flat: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        out_parts = []
        start = 0
        for K in self.num_classes:
            sl = slice(start, start+K)
            probs = F.softmax(logits_flat[:, sl] / temperature, dim=1)
            idx = torch.multinomial(probs, 1).squeeze(1)
            one_hot = F.one_hot(idx, num_classes=K).float()
            out_parts.append((one_hot + 1e-30).log())
            start += K
        return torch.cat(out_parts, dim=1)


    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred_eps: torch.Tensor) -> torch.Tensor:
        """
        Pomocnicza funkcja do odzyskania x0 z xt i przewidzianego szumu (dla części numerycznej).
        """
        sqrt_recip_alphas_bar = extract(1.0 / self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_recip_m1_alphas_bar = extract(torch.sqrt(1.0 / self.alphas_bar - 1.0), t, x_t.shape)
        pred_x0 = sqrt_recip_alphas_bar * x_t - sqrt_recip_m1_alphas_bar * pred_eps
        return pred_x0.clamp(-5.0, 5.0)


    def forward(self, x_neigh: torch.Tensor, x_orig: torch.Tensor, y_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        B = x_neigh.shape[0]
        t = torch.randint(0, len(self.betas), (B,), device=x_neigh.device)
        x_num = x_neigh[:, :self.num_numerical]
        x_cat_log = x_neigh[:, self.num_numerical:]

        noise_num = torch.randn_like(x_num)
        x_num_t = self.q_sample_gauss(x_num, t, noise_num)
        if self.total_cat_dim > 0:
            x_cat_t_log = self.q_sample_cat(x_cat_log, t)
        else:
            x_cat_t_log = x_cat_log
        x_in_t = torch.cat([x_num_t, x_cat_t_log], dim=1)
        model_out = self.denoise_fn(x_in_t, t, x_orig, y_target)

        pred_num = model_out[:, :self.num_numerical]
        pred_cat_logits = model_out[:, self.num_numerical:]
        loss_num = F.mse_loss(pred_num, noise_num)

        loss_cat = torch.tensor(0.0, device=x_neigh.device)
        if self.total_cat_dim > 0:
            start = 0
            losses = []
            for K in self.num_classes:
                target_idx = x_cat_log[:, start:start+K].argmax(dim=1)
                logits = pred_cat_logits[:, start:start+K]
                losses.append(F.cross_entropy(logits, target_idx))
                start += K
            loss_cat = torch.stack(losses).mean()

        total_loss = loss_num + loss_cat
        return total_loss, {"num": loss_num.item(), "cat": loss_cat.item()}

    @torch.no_grad()
    def sample_counterfactual(self, x_orig: torch.Tensor, y_target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        B = x_orig.shape[0]
        device = x_orig.device
        x_num = torch.randn(B, self.num_numerical, device=device) * temperature
        
        cat_parts = []
        for K in self.num_classes:
            cat_parts.append(torch.zeros(B, K, device=device))
        x_cat_log = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros(B, 0, device=device)

        for i in reversed(range(0, len(self.betas))):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            x_in = torch.cat([x_num, x_cat_log], dim=1)
            
            model_out = self.denoise_fn(x_in, t, x_orig, y_target)
            pred_eps_num = model_out[:, :self.num_numerical]
            pred_logits_cat = model_out[:, self.num_numerical:]
            
            mean, log_var = self.p_mean_variance_gauss(x_num, t, pred_eps_num)
            noise = torch.randn_like(x_num) if i > 0 else 0.0
            x_num = mean + torch.exp(0.5 * log_var) * noise * temperature
            if self.total_cat_dim > 0:
                x_cat_log = self.p_sample_cat(x_cat_log, t, pred_logits_cat, temperature=temperature)

        if self.total_cat_dim > 0:
             x_cat_final = torch.exp(x_cat_log)
             return torch.cat([x_num, x_cat_final], dim=1)
             
        return x_num

    @torch.no_grad()
    def sample_with_svdd(
        self, 
        x_orig: torch.Tensor, 
        y_target: torch.Tensor, 
        clf_model: nn.Module, 
        num_candidates: int = 10, 
        guidance_scale: float = 20.0,
        dist_scale: float = 0.1, 
        cat_scale: float = 1.0,
        temperature: float = 1.0
    ) -> torch.Tensor:

        B = x_orig.shape[0]
        device = x_orig.device
        M = num_candidates

        x_orig_expanded = x_orig.repeat_interleave(M, dim=0)
        y_target_expanded = y_target.repeat_interleave(M, dim=0)
        x_num = torch.randn(B * M, self.num_numerical, device=device) * temperature
        
        cat_parts = []
        for K in self.num_classes:
            cat_parts.append(torch.zeros(B * M, K, device=device))
        x_cat_log = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros(B * M, 0, device=device)

        for i in reversed(range(0, len(self.betas))):
            t = torch.full((B * M,), i, device=device, dtype=torch.long)
            
            x_in = torch.cat([x_num, x_cat_log], dim=1)
            model_out = self.denoise_fn(x_in, t, x_orig_expanded, y_target_expanded)
            
            pred_eps_num = model_out[:, :self.num_numerical]
            pred_logits_cat = model_out[:, self.num_numerical:]
            
            if i > 0:
                pred_x0_num = self._predict_x0_from_eps(x_num, t, pred_eps_num)
                pred_x0_cat_probs_list = []
                cat_changes_count = torch.zeros(B * M, device=device)
                
                start_logits = 0
                start_orig = self.num_numerical
                
                for K in self.num_classes:
                    logits_k = pred_logits_cat[:, start_logits : start_logits + K]
                    probs_k = F.softmax(logits_k, dim=1)
                    pred_x0_cat_probs_list.append(probs_k)
                    
                    pred_cat_idx = torch.argmax(logits_k, dim=1)
                    
                    orig_segment = x_orig_expanded[:, start_orig : start_orig + K]
                    orig_cat_idx = torch.argmax(orig_segment, dim=1)
                    
                    is_different = (pred_cat_idx != orig_cat_idx).float()
                    cat_changes_count += is_different
                    
                    start_logits += K
                    start_orig += K
                
                if pred_x0_cat_probs_list:
                    pred_x0_cat = torch.cat(pred_x0_cat_probs_list, dim=1)
                else:
                    pred_x0_cat = torch.tensor([], device=device)
                
                x0_estimate = torch.cat([pred_x0_num, pred_x0_cat], dim=1)
                clf_logits = clf_model(x0_estimate)
                clf_probs = F.softmax(clf_logits, dim=1)
                validity_reward = clf_probs.gather(1, y_target_expanded.unsqueeze(1)).squeeze(1)
                
                orig_num = x_orig_expanded[:, :self.num_numerical]
                dist_num_sq = torch.sum((pred_x0_num - orig_num) ** 2, dim=1)
                
                total_reward = validity_reward \
                               - (dist_scale * dist_num_sq) \
                               - (cat_scale * cat_changes_count) 
                
                weights = F.softmax(total_reward.view(B, M) * guidance_scale, dim=1)
                selected_indices = torch.multinomial(weights, 1).squeeze(1)
                
                batch_offsets = torch.arange(B, device=device) * M
                global_indices = batch_offsets + selected_indices
                
                x_num = x_num[global_indices].repeat_interleave(M, dim=0)
                x_cat_log = x_cat_log[global_indices].repeat_interleave(M, dim=0)
                pred_eps_num = pred_eps_num[global_indices].repeat_interleave(M, dim=0)
                pred_logits_cat = pred_logits_cat[global_indices].repeat_interleave(M, dim=0)

            mean, log_var = self.p_mean_variance_gauss(x_num, t, pred_eps_num)
            noise = torch.randn_like(x_num) if i > 0 else 0.0
            x_num = mean + torch.exp(0.5 * log_var) * noise * temperature
            
            if self.total_cat_dim > 0:
                x_cat_log = self.p_sample_cat(x_cat_log, t, pred_logits_cat, temperature=temperature)

        if self.total_cat_dim > 0:
             x_cat_final = torch.exp(x_cat_log)
             return torch.cat([x_num, x_cat_final], dim=1)
             
        return x_num