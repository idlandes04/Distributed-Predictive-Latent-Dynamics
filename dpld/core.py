import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math # For checking nan/inf

from utils import estimate_lyapunov_exponent, sparsify_vector

# --- Constants ---
EPSILON = 1e-8 # For numerical stability
DEFAULT_ACTION_STD = 0.05 # Reduced default action noise
CLS_NORM_CLIP_VAL = 100.0 # Value for optional CLS norm clipping
MODULE_OUTPUT_CLIP_VAL = 100.0 # Clip raw module predictions
ACTION_MEAN_CLIP_VAL = 1000.0 # Clip the mean of the write action distribution


# --- Predictive Module ---
class PredictiveModule(nn.Module):
    """
    Implements a DPLD module: Read, Predict, Write (via sparse, gated contribution).
    Learns via difference reward based on global surprise.
    Includes stability clamping.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD, device='cpu'): # Added action_std parameter
        super().__init__()
        self.cls_dim = cls_dim
        self.k_sparse_write = k_sparse_write
        self.device = device
        self.surprise_scale_factor = surprise_scale_factor
        self.surprise_baseline_ema = surprise_baseline_ema
        self.action_std = action_std # Store action noise level

        # Internal Predictive Model (fm)
        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)

        # Gating Query Vector (qm)
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01) # Reduced initial scale

        self.register_buffer('sm_baseline', torch.tensor(1.0, device=device))
        self.last_prediction_ct_plus_1 = None
        self.last_surprise_sm = None
        self.last_log_prob = None

        # Combine parameters for optimizer
        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate)


    def predict(self, ct):
        """Predicts the next CLS state, clamping the output."""
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        prediction = self.fm(ct_dense.detach())
        # --- Stability: Clip module output ---
        prediction = torch.clamp(prediction, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_prediction_ct_plus_1 = prediction
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1):
        """Calculates local surprise Sm based on prediction and actual next state."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             print("Warning: calculate_surprise called before predict. Returning baseline surprise.")
             return self.sm_baseline.detach()

        # Ensure prediction is finite before loss calculation
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            print("Warning: Non-finite prediction encountered in calculate_surprise. Using baseline.")
            self.last_prediction_ct_plus_1 = torch.zeros_like(self.last_prediction_ct_plus_1) # Reset prediction
            surprise = self.sm_baseline.detach() * 10 # Penalize? Or just use baseline?
        else:
            surprise = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')

        # Ensure surprise is finite before updating baseline
        if not math.isfinite(surprise.item()):
            print(f"Warning: Non-finite surprise calculated ({surprise.item()}). Skipping baseline update.")
            # Use detached baseline as the current surprise to avoid propagating NaN/inf
            self.last_surprise_sm = self.sm_baseline.detach()
        else:
            self.last_surprise_sm = surprise
            with torch.no_grad():
                self.sm_baseline = self.surprise_baseline_ema * self.sm_baseline + \
                                   (1 - self.surprise_baseline_ema) * self.last_surprise_sm
                self.sm_baseline = torch.clamp(self.sm_baseline, min=1e-6, max=1e6)

        return self.last_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse, weighted, gated write vector Im, with stability checks."""
        if self.last_prediction_ct_plus_1 is None:
             raise RuntimeError("Must call predict() before generate_write_vector()")

        current_surprise_for_alpha = self.sm_baseline.detach()

        # Alg 1 Steps 1-5: Calculate intermediate dense write vector
        vm = self.last_prediction_ct_plus_1 # Already clamped in predict()
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        tau_g = 1.0
        gate_activation_gm = torch.sigmoid(self.qm * ct_dense.detach() / tau_g)
        alpha_base = 1.0
        alpha_scale = 1.0
        surprise_diff = current_surprise_for_alpha - self.sm_baseline
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)

        # --- Stochasticity & Stability ---
        action_mean = intermediate_write_vector

        # --- Stability: Check and clamp action_mean before Normal distribution ---
        if not torch.all(torch.isfinite(action_mean)):
            print(f"Warning: NaN/inf detected in action_mean for module. Clamping.")
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=ACTION_MEAN_CLIP_VAL, neginf=-ACTION_MEAN_CLIP_VAL)
        action_mean = torch.clamp(action_mean, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

        action_std_tensor = torch.full_like(action_mean, fill_value=self.action_std)

        # Create distribution and sample
        try:
            dist = Normal(action_mean, action_std_tensor)
            dense_write_vector_sampled = dist.sample()
            # Calculate log_prob using detached sample
            self.last_log_prob = dist.log_prob(dense_write_vector_sampled.detach()).sum()
        except ValueError as e:
             print(f"ERROR creating Normal distribution: {e}")
             print(f"action_mean stats: min={action_mean.min()}, max={action_mean.max()}, has_nan={torch.isnan(action_mean).any()}")
             print(f"action_std_tensor stats: min={action_std_tensor.min()}, max={action_std_tensor.max()}, has_nan={torch.isnan(action_std_tensor).any()}")
             # Fallback: return zero vector and zero log_prob
             dense_write_vector_sampled = torch.zeros_like(action_mean)
             self.last_log_prob = torch.tensor(0.0, device=self.device) # Zero log prob for safety


        # Alg 1 Steps 6-7: Sparsify
        write_vector_im_sparse_vals = sparsify_vector(dense_write_vector_sampled, self.k_sparse_write)
        sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device)
        else:
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=torch.float32, device=self.device),
                                                 (self.cls_dim,))

        return im_sparse.coalesce()

    def learn(self, difference_reward_rm):
        """Updates module parameters using the difference reward."""
        if self.last_log_prob is None or self.last_log_prob == 0.0: # Check for fallback case
             return 0.0

        # Ensure reward is finite, otherwise skip update
        if not math.isfinite(difference_reward_rm):
             print(f"Warning: Non-finite difference reward ({difference_reward_rm}). Skipping module learn step.")
             self.last_log_prob = None
             return 0.0

        # Use .clone().detach() to avoid the UserWarning if difference_reward_rm is already a tensor
        reward_tensor = difference_reward_rm.clone().detach() if isinstance(difference_reward_rm, torch.Tensor) else torch.tensor(difference_reward_rm, device=self.device)

        loss = -reward_tensor * self.last_log_prob

        # --- Stability: Check loss before backward ---
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss ({loss.item()}) calculated in module learn. Skipping backward pass.")
            self.last_log_prob = None
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        # --- Stability: Gradient clipping ---
        nn.utils.clip_grad_norm_(list(self.fm.parameters()) + [self.qm], max_norm=1.0)
        self.optimizer.step()

        self.last_log_prob = None
        self.last_prediction_ct_plus_1 = None

        return loss.item()


# --- Meta-Model ---
# (No changes needed from previous version, but ensure gamma range in train.py is updated)
class MetaModel(nn.Module):
    """
    Implements a simplified DPLD Meta-Model for stability regulation.
    Monitors CLS dynamics (lambda_max) and adjusts global decay (gamma_t).
    NOTE: Learning mechanism is disabled in this MVP version.
    """
    def __init__(self, cls_dim, meta_hidden_dim, learning_rate,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.device = device
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.stability_target = stability_target # lambda_thr in Thm 7.1

        # Controller network (currently unused for learning)
        self.controller = nn.Sequential(
            nn.Linear(cls_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, 1) # Output: single value controlling gamma
        ).to(device)
        # Parameters θMM are implicitly self.controller.parameters()

        # Optimizer (currently unused)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Store the last estimated lambda_max
        self.last_estimated_lambda_max = None


    def update_stability_estimate(self, lambda_max_estimate):
        """Stores the latest LE estimate."""
        self.last_estimated_lambda_max = lambda_max_estimate

    def compute_regulation(self, current_ct):
        """
        Computes adaptive decay gamma_t based on the last LE estimate.
        Computes modulatory vector mmod_t (zero for MVP).
        """
        # --- Adaptive Decay γt ---
        # Use the last estimated lambda_max (potentially from a previous step)
        if self.last_estimated_lambda_max is not None and np.isfinite(self.last_estimated_lambda_max):
             # Sigmoid mapping from lambda_max to gamma range
             # Increase sensitivity: scale difference before sigmoid
             sensitivity = 5.0
             gamma_signal = torch.sigmoid(torch.tensor(sensitivity * (self.last_estimated_lambda_max - self.stability_target), device=self.device))
             gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * gamma_signal
             gamma_t = torch.clamp(gamma_t, self.gamma_min, self.gamma_max) # Ensure bounds
             gamma_t = gamma_t.item() # Convert to scalar float
        else:
             # Default gamma if stability not estimated yet or estimate was invalid
             # Start with a higher default gamma for stability
             gamma_t = self.gamma_max # (self.gamma_min + self.gamma_max) / 2.0

        # --- Modulatory Vector mmod_t ---
        # Set to zero sparse tensor for MVP
        mmod_t = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                         torch.empty((0,), dtype=torch.float32, device=self.device),
                                         (self.cls_dim,))

        return gamma_t, mmod_t


    def learn(self):
        """Updates Meta-Model parameters (DISABLED in MVP)."""
        if self.last_estimated_lambda_max is None or not np.isfinite(self.last_estimated_lambda_max):
            return 0.0 # Cannot learn without stability estimate

        instability_penalty = F.relu(torch.tensor(self.last_estimated_lambda_max - self.stability_target, device=self.device))
        loss = instability_penalty # Simple ReLU penalty

        return loss.item()


# --- DPLD System ---
class DPLDSystem(nn.Module):
    """
    The main DPLD system coordinating CLS, Modules, and Meta-Model.
    Includes optional CLS norm clipping.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, noise_std_dev_schedule,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1,
                 action_std=DEFAULT_ACTION_STD, # Pass down action_std
                 clip_cls_norm=True, # Add option to clip CLS norm
                 device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules
        self.k_sparse_write = k_sparse_write
        self.noise_std_dev_schedule = noise_std_dev_schedule
        self.clip_cls_norm = clip_cls_norm
        self.device = device

        # Initialize CLS state (sparse tensor)
        self.ct = self._init_cls()

        # Initialize Modules
        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                             action_std=action_std, device=device) # Pass action_std
            for _ in range(num_modules)
        ])

        # Initialize Meta-Model
        self.meta_model = MetaModel(cls_dim, meta_hidden_dim, meta_lr,
                                    gamma_min, gamma_max, stability_target, device=device)

        self.last_global_surprise_gt = None

    def _init_cls(self):
        # Start with smaller magnitude
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.01
        # Ensure initial state is actually sparse
        sparse_ct = sparsify_vector(initial_dense, self.k_sparse_write).to_sparse_coo()
        return sparse_ct.coalesce()


    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation with optional norm clipping."""
        if not ct.is_sparse: ct = ct.to_sparse_coo()
        if not sum_im.is_sparse: sum_im = sum_im.to_sparse_coo()
        if not mmodt.is_sparse: mmodt = mmodt.to_sparse_coo()

        decayed_ct = (1.0 - gamma_t) * ct
        noise_et = torch.randn(self.cls_dim, device=self.device) * noise_std_dev

        # Combine sparse terms first
        ct_plus_1_sparse = (decayed_ct + sum_im + mmodt).coalesce()

        # Add dense noise
        ct_plus_1_dense = ct_plus_1_sparse.to_dense() + noise_et

        # --- Stability: Optional CLS norm clipping ---
        if self.clip_cls_norm:
            norm = torch.linalg.norm(ct_plus_1_dense)
            if norm > CLS_NORM_CLIP_VAL:
                 # print(f"Clipping CLS norm: {norm:.2f} -> {CLS_NORM_CLIP_VAL}") # Debug
                 ct_plus_1_dense = ct_plus_1_dense * (CLS_NORM_CLIP_VAL / (norm + EPSILON))

        # Convert back to sparse - Note: density might still be high due to noise
        # Consider re-sparsifying here if strict density is needed, but adds complexity.
        ct_plus_1 = ct_plus_1_dense.to_sparse_coo()

        return ct_plus_1.coalesce()


    def get_dynamics_map(self, fixed_gamma, fixed_noise_std=0.0):
         """
         Returns a function representing the one-step CLS dynamics for LE estimation.
         Uses current module parameters but *fixed* gamma and noise for consistency.
         Includes stability clamping within the map definition.
         """
         gamma_val = float(fixed_gamma)

         def dynamics_map(state_t_dense):
             # Ensure input state is finite for JVP
             if not torch.all(torch.isfinite(state_t_dense)):
                 print("Error: Non-finite input state to dynamics_map for LE.")
                 # Return a zero tensor or raise error? Returning zero might hide issues.
                 # Let's return the input, LE estimator might handle it.
                 return state_t_dense

             state_t_sparse = state_t_dense.to_sparse_coo()
             sum_im = self._init_cls()

             with torch.no_grad():
                 for i, module in enumerate(self.pred_modules):
                     # --- Replicate prediction and deterministic write vector generation ---
                     # Predict & Clamp
                     pred = module.fm(state_t_sparse.to_dense().detach())
                     pred = torch.clamp(pred, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)

                     # Use baseline surprise for alpha_m
                     current_surprise_for_alpha = module.sm_baseline.detach()

                     # Calculate deterministic intermediate write vector
                     vm = pred # Use clamped prediction
                     ct_dense_detached = state_t_sparse.to_dense().detach()
                     tau_g = 1.0
                     gate_activation_gm = torch.sigmoid(module.qm.detach() * ct_dense_detached / tau_g)
                     alpha_base = 1.0; alpha_scale = 1.0
                     surprise_diff = current_surprise_for_alpha - module.sm_baseline.detach()
                     influence_scalar_am = alpha_base + alpha_scale * torch.tanh(module.surprise_scale_factor * surprise_diff)
                     influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                     intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)

                     # Clamp the mean action
                     intermediate_write_vector = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

                     # Sparsify the *mean* action
                     write_vector_im_sparse_vals = sparsify_vector(intermediate_write_vector, module.k_sparse_write)
                     sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
                     sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                     if sparse_indices.numel() > 0:
                         im = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device)
                     else:
                         im = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                      torch.empty((0,), dtype=torch.float32, device=self.device),
                                                      (self.cls_dim,))
                     sum_im += im
                 # --- End Module Loop ---

                 gamma_t = gamma_val
                 mmodt = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                  torch.empty((0,), dtype=torch.float32, device=self.device),
                                                  (self.cls_dim,))

                 next_state_sparse = self.cls_update_rule(state_t_sparse, sum_im.coalesce(), mmodt, gamma_t, fixed_noise_std)
                 next_state_dense = next_state_sparse.to_dense()

                 # Ensure output is finite
                 if not torch.all(torch.isfinite(next_state_dense)):
                     print("Warning: Non-finite output state from dynamics_map for LE. Clamping.")
                     next_state_dense = torch.nan_to_num(next_state_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)


             return next_state_dense # Return dense for JVP

         return dynamics_map


    def step(self, current_step_num, estimate_le=False):
        """Performs one full step of the DPLD system interaction with stability checks."""

        # --- Check current state ---
        if not torch.all(torch.isfinite(self.ct.values())):
             print(f"ERROR: CLS state became non-finite at step {current_step_num}. Resetting state.")
             self.ct = self._init_cls() # Reset CLS
             self.last_global_surprise_gt = None # Reset GT baseline
             # Optionally reset module baselines?
             for module in self.pred_modules:
                 module.sm_baseline.fill_(1.0)

        ct_prev_dense = self.ct.to_dense().detach()

        # --- 1. Module Predictions ---
        predictions = []
        for module in self.pred_modules:
            pred = module.predict(self.ct.detach())
            predictions.append(pred)

        # --- 2. Meta-Model Regulation ---
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.ct.detach())

        # --- 3. Module Write Vectors ---
        write_vectors_im = []
        sum_im = self._init_cls()
        for i, module in enumerate(self.pred_modules):
             im = module.generate_write_vector(self.ct.detach())
             write_vectors_im.append(im)
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual Surprises & Global Surprise ---
        surprises_sm = []
        global_surprise_gt_sum = 0.0
        valid_surprises = 0
        for i, module in enumerate(self.pred_modules):
             sm = module.calculate_surprise(ct_plus_1)
             if math.isfinite(sm.item()):
                 surprises_sm.append(sm)
                 global_surprise_gt_sum += sm
                 valid_surprises += 1
             else:
                 surprises_sm.append(torch.tensor(float('nan'), device=self.device)) # Keep placeholder

        global_surprise_gt = (global_surprise_gt_sum / valid_surprises) if valid_surprises > 0 else torch.tensor(0.0, device=self.device)

        # --- 6. Calculate Difference Rewards ---
        difference_rewards_rm = []
        if self.num_modules > 0 and self.last_global_surprise_gt is not None and math.isfinite(self.last_global_surprise_gt):
            # Calculate Gt^{-m} only if Gt is valid
            if math.isfinite(global_surprise_gt.item()):
                for m_idx in range(self.num_modules):
                    sum_im_counterfactual = self._init_cls()
                    for i in range(self.num_modules):
                        if i != m_idx and write_vectors_im[i].is_sparse:
                             sum_im_counterfactual += write_vectors_im[i]
                    sum_im_counterfactual = sum_im_counterfactual.coalesce()

                    ct_plus_1_counterfactual = self.cls_update_rule(
                        self.ct, sum_im_counterfactual, mmod_t, gamma_t, noise_std_dev
                    )
                    ct_plus_1_cf_dense = ct_plus_1_counterfactual.to_dense().detach()

                    gt_counterfactual_sum = 0.0
                    valid_cf_surprises = 0
                    for i, module in enumerate(self.pred_modules):
                        if predictions[i] is not None and torch.all(torch.isfinite(predictions[i])):
                             sm_counterfactual = F.mse_loss(predictions[i].detach(), ct_plus_1_cf_dense, reduction='mean')
                             if math.isfinite(sm_counterfactual.item()):
                                 gt_counterfactual_sum += sm_counterfactual
                                 valid_cf_surprises += 1
                    gt_counterfactual = (gt_counterfactual_sum / valid_cf_surprises) if valid_cf_surprises > 0 else torch.tensor(0.0, device=self.device)

                    rm = gt_counterfactual - global_surprise_gt
                    difference_rewards_rm.append(rm.detach())
            else: # Gt was not finite
                 difference_rewards_rm = [torch.tensor(0.0, device=self.device) for _ in range(self.num_modules)]
        else: # First step or invalid last_gt
            difference_rewards_rm = [torch.tensor(0.0, device=self.device) for _ in range(self.num_modules)]

        # --- 7. Update State and Store Previous GT ---
        self.ct = ct_plus_1
        self.last_global_surprise_gt = global_surprise_gt.item() if math.isfinite(global_surprise_gt.item()) else None


        # --- 8. Trigger Learning ---
        module_losses = []
        for i, module in enumerate(self.pred_modules):
            reward = difference_rewards_rm[i] if i < len(difference_rewards_rm) else 0.0
            loss = module.learn(reward)
            module_losses.append(loss)

        meta_loss = self.meta_model.learn()

        # --- 9. Optional: Estimate LE ---
        lambda_max_estimate = None
        if estimate_le:
            # Only estimate if previous state was finite
            if torch.all(torch.isfinite(ct_prev_dense)):
                dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
                lambda_max_estimate = estimate_lyapunov_exponent(
                    dynamics_map_for_le,
                    ct_prev_dense,
                    device=self.device
                )
                if lambda_max_estimate is not None and not math.isfinite(lambda_max_estimate):
                     print(f"Warning: Non-finite LE estimate ({lambda_max_estimate}). Discarding.")
                     lambda_max_estimate = None # Discard invalid estimate
                self.meta_model.update_stability_estimate(lambda_max_estimate)
            else:
                print("Skipping LE estimation due to non-finite previous state.")


        # --- 10. Return Metrics ---
        # Calculate metrics safely, handling potential NaNs
        finite_surprises = [s.item() for s in surprises_sm if s is not None and math.isfinite(s.item())]
        finite_rewards = [r.item() for r in difference_rewards_rm if r is not None and math.isfinite(r.item())]
        finite_losses = [l for l in module_losses if l is not None and math.isfinite(l)]

        metrics = {
            "Gt": global_surprise_gt.item() if math.isfinite(global_surprise_gt.item()) else float('nan'),
            "Sm_avg": np.mean(finite_surprises) if finite_surprises else float('nan'),
            "Sm_std": np.std(finite_surprises) if len(finite_surprises) > 1 else 0.0,
            "Rm_avg": np.mean(finite_rewards) if finite_rewards else float('nan'),
            "Rm_std": np.std(finite_rewards) if len(finite_rewards) > 1 else 0.0,
            "lambda_max_est": lambda_max_estimate,
            "gamma_t": gamma_t,
            "noise_std": noise_std_dev,
            "module_loss_avg": np.mean(finite_losses) if finite_losses else float('nan'),
            "meta_loss": meta_loss,
            "cls_norm": torch.linalg.norm(self.ct.to_dense()).item() if torch.all(torch.isfinite(self.ct.values())) else float('nan'),
             # Note: cls_density might be high due to dense noise addition
            "cls_density": self.ct._nnz() / self.cls_dim if self.ct._nnz() is not None else float('nan')
        }

        return metrics