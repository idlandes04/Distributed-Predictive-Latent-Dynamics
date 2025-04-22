# dpld/core.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math # For checking nan/inf

from utils import estimate_lyapunov_exponent, sparsify_vector # Assuming these are in utils.py

# --- Constants (Revised Action Std) ---
EPSILON = 1e-8 # For numerical stability
DEFAULT_ACTION_STD = 0.5 # Increased default action std dev
CLS_NORM_CLIP_VAL = 100.0
MODULE_OUTPUT_CLIP_VAL = 100.0
ACTION_MEAN_CLIP_VAL = 1000.0
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping
META_REWARD_CLIP = 10.0 # Clipping range for meta-model reward signal


# --- Math Encoder Module (Unchanged) ---
class MathEncoder(nn.Module):
    """Encodes an arithmetic problem (a, op, b) into a sparse CLS vector."""
    def __init__(self, num_vocab_size, op_vocab_size, embedding_dim, cls_dim, k_sparse_write, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.k_sparse_write = k_sparse_write
        self.device = device

        self.num_embedding = nn.Embedding(num_vocab_size, embedding_dim)
        self.op_embedding = nn.Embedding(op_vocab_size, embedding_dim)

        total_embedding_dim = embedding_dim * 3 # a, op, b
        self.w_enc = nn.Linear(total_embedding_dim, cls_dim)

    def forward(self, a, op_idx, b):
        a = a.to(self.device)
        op_idx = op_idx.to(self.device)
        b = b.to(self.device)
        a_embed = self.num_embedding(a)
        op_embed = self.op_embedding(op_idx)
        b_embed = self.num_embedding(b)
        combined_embed = torch.cat([a_embed, op_embed, b_embed], dim=-1)
        projected = self.w_enc(combined_embed)
        sparse_write_vals = sparsify_vector(projected, self.k_sparse_write)
        sparse_indices = torch.where(sparse_write_vals != 0)[0].unsqueeze(0)
        sparse_values = sparse_write_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             i_math = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=projected.dtype)
        else:
             i_math = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                torch.empty((0,), dtype=projected.dtype, device=self.device),
                                                (self.cls_dim,))
        return i_math.coalesce()


# --- Predictive Module (Base Class - Revised for Weight Decay) ---
class PredictiveModule(nn.Module):
    """
    Base class for DPLD modules: Read, Predict, Write.
    Learns via normalized difference reward + entropy bonus.
    Uses LOG-SURPRISE internally and for influence scaling.
    Includes stability clamping and weight decay.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0, # Scaling factor for surprise diff in alpha_m
                 surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD, task_loss_weight=0.0,
                 weight_decay=0.0, # Added weight decay
                 device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.module_hidden_dim = module_hidden_dim
        self.k_sparse_write = k_sparse_write
        self.device = device
        self.entropy_coeff = entropy_coeff
        self.surprise_scale_factor = surprise_scale_factor
        self.surprise_baseline_ema = surprise_baseline_ema
        self.action_std = action_std
        self.task_loss_weight = task_loss_weight

        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)

        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01)

        # Baseline is now for LOG-SURPRISE
        self.register_buffer('sm_log_baseline', torch.tensor(0.0, device=device)) # Initialize baseline for log-surprise
        self.last_prediction_ct_plus_1 = None
        self.last_log_surprise_sm = None      # Stores combined LOG-surprise
        self.last_log_surprise_sm_cls = None  # Stores LOG-surprise for CLS part
        self.last_log_surprise_sm_task = None # Stores LOG-surprise for task part
        self.last_raw_surprise_sm_cls = None  # Store raw value for logging
        self.last_raw_surprise_sm_task = None # Store raw value for logging
        self.last_log_prob = None
        self.last_action_dist = None

        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate, weight_decay=weight_decay) # Added weight_decay

    def predict(self, ct):
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        prediction = self.fm(ct_dense.detach())
        prediction = torch.clamp(prediction, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_prediction_ct_plus_1 = prediction
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates local LOG-SURPRISE Sm."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             self.last_log_surprise_sm = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device)
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             self.last_raw_surprise_sm_task = torch.tensor(float('nan'), device=self.device)
             return self.last_log_surprise_sm

        # --- Calculate RAW Sm_cls ---
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            sm_cls_raw = torch.tensor(1e6, device=self.device) # Assign high raw surprise if prediction invalid
        else:
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach()

        # --- Calculate RAW Sm_task ---
        sm_task_raw = torch.tensor(0.0, device=self.device) # Default for base module
        self.last_raw_surprise_sm_task = sm_task_raw.detach()

        # --- Apply Log Transform (log1p for stability near zero) ---
        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)
        sm_task_log = torch.log1p(sm_task_raw + EPSILON)

        self.last_log_surprise_sm_cls = sm_cls_log.detach()
        self.last_log_surprise_sm_task = sm_task_log.detach()

        # --- Combine LOG-Surprises ---
        combined_log_surprise = sm_cls_log + self.task_loss_weight * sm_task_log

        # Ensure combined log surprise is finite before updating baseline
        if not math.isfinite(combined_log_surprise.item()):
            self.last_log_surprise_sm = self.sm_log_baseline.detach()
        else:
            self.last_log_surprise_sm = combined_log_surprise
            with torch.no_grad():
                # Update baseline using LOG surprise
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * self.last_log_surprise_sm

        return self.last_log_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse write vector Im using LOG-SURPRISE for alpha_m."""
        if self.last_prediction_ct_plus_1 is None:
             raise RuntimeError(f"{self.__class__.__name__}: Must call predict() before generate_write_vector()")

        current_log_surprise = self.last_log_surprise_sm.detach() if self.last_log_surprise_sm is not None else self.sm_log_baseline.detach()
        log_surprise_diff = current_log_surprise - self.sm_log_baseline.detach()

        vm = self.last_prediction_ct_plus_1
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        tau_g = 1.0
        gate_raw_score = self.qm * ct_dense.detach()
        gate_activation_gm = torch.sigmoid(gate_raw_score / tau_g)
        alpha_base = 1.0
        alpha_scale = 1.0
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * log_surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
        action_mean = intermediate_write_vector

        if not torch.all(torch.isfinite(action_mean)):
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=ACTION_MEAN_CLIP_VAL, neginf=-ACTION_MEAN_CLIP_VAL)
        action_mean = torch.clamp(action_mean, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

        # Use the potentially increased action_std
        action_std_tensor = torch.full_like(action_mean, fill_value=self.action_std)

        try:
            dist = Normal(action_mean, action_std_tensor)
            self.last_action_dist = dist # Store distribution for entropy
            dense_write_vector_sampled = dist.sample()
            self.last_log_prob = dist.log_prob(dense_write_vector_sampled).sum() # Use the sampled action
        except ValueError as e:
             print(f"ERROR creating Normal distribution in {self.__class__.__name__}: {e}")
             dense_write_vector_sampled = torch.zeros_like(action_mean)
             self.last_log_prob = torch.tensor(0.0, device=self.device)
             self.last_action_dist = None

        write_vector_im_sparse_vals = sparsify_vector(dense_write_vector_sampled, self.k_sparse_write)
        sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=dense_write_vector_sampled.dtype)
        else:
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=dense_write_vector_sampled.dtype, device=self.device),
                                                 (self.cls_dim,))
        return im_sparse.coalesce()

    def learn(self, normalized_difference_reward_rm):
        """Updates module parameters using normalized difference reward and entropy bonus."""
        entropy_tensor = torch.tensor(0.0, device=self.device) # Initialize as tensor
        entropy_item = 0.0
        grad_norm = 0.0

        if self.last_log_prob is None or self.last_log_prob == 0.0:
             return 0.0, 0.0, entropy_item, grad_norm

        if not math.isfinite(normalized_difference_reward_rm):
             self.last_log_prob = None
             self.last_action_dist = None
             return 0.0, 0.0, entropy_item, grad_norm

        if self.last_action_dist is not None:
            try:
                entropy_tensor = self.last_action_dist.entropy().sum() # Keep as tensor initially
                if not torch.isfinite(entropy_tensor):
                    entropy_tensor = torch.tensor(0.0, device=self.device)
                entropy_item = entropy_tensor.item() # Get item for logging later
            except Exception as e:
                 print(f"Error calculating entropy in {self.__class__.__name__}: {e}")
                 entropy_tensor = torch.tensor(0.0, device=self.device)
                 entropy_item = 0.0

        entropy_term = self.entropy_coeff * entropy_tensor # Use tensor in loss calculation

        reward_tensor = torch.tensor(normalized_difference_reward_rm, device=self.device)
        if not torch.isfinite(self.last_log_prob):
            self.last_log_prob = None
            self.last_action_dist = None
            return 0.0, 0.0, entropy_item, grad_norm

        policy_loss = -(reward_tensor * self.last_log_prob + entropy_term) # Minimize -(Reward*log_prob + H)

        if not torch.isfinite(policy_loss):
            self.last_log_prob = None
            self.last_action_dist = None
            return 0.0, 0.0, entropy_item, grad_norm

        self.optimizer.zero_grad()
        policy_loss.backward()

        params_to_clip = list(self.fm.parameters()) + [self.qm]
        total_norm_sq = 0.0
        for p in params_to_clip:
            if p.grad is not None:
                param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
        grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

        nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        policy_loss_item = policy_loss.item()
        task_loss_item = self.last_log_surprise_sm_task.item() if self.last_log_surprise_sm_task is not None else 0.0

        self.last_log_prob = None
        self.last_action_dist = None

        return policy_loss_item, task_loss_item, entropy_item, grad_norm


# --- Task Head Module (Arithmetic Prediction - Revised for Weight Decay) ---
class TaskHead(PredictiveModule):
    """
    Specialized PredictiveModule for the arithmetic task.
    Uses LOG-SURPRISE. Internal model `fm` uses a GRU.
    Includes weight decay.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD, task_loss_weight=1.0, # Default weight = 1.0
                 weight_decay=0.0, # Added weight decay
                 gru_layers=1, device='cpu'):
        super().__init__(cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                         entropy_coeff, surprise_scale_factor, surprise_baseline_ema, action_std,
                         task_loss_weight, weight_decay, device) # Pass weight_decay

        self.gru = nn.GRU(cls_dim, module_hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.task_out_layer = nn.Linear(module_hidden_dim, 1).to(device)
        self.cls_pred_layer = nn.Linear(module_hidden_dim, cls_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.gru.parameters()) +
            list(self.task_out_layer.parameters()) +
            list(self.cls_pred_layer.parameters()) +
            [self.qm],
            lr=learning_rate,
            weight_decay=weight_decay # Added weight_decay
        )

        self.last_gru_hidden_state = None
        self.last_task_prediction = None

    def predict(self, ct):
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        ct_input = ct_dense.detach().unsqueeze(0).unsqueeze(0)
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None

        if hidden_state_input is not None and hidden_state_input.shape[1] != ct_input.shape[0]:
             hidden_state_input = None

        gru_output, self.last_gru_hidden_state = self.gru(ct_input, hidden_state_input)
        last_hidden = gru_output[:, -1, :]
        predicted_answer = self.task_out_layer(last_hidden).squeeze()
        self.last_task_prediction = torch.clamp(predicted_answer, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        predicted_cls = self.cls_pred_layer(last_hidden).squeeze(0)
        self.last_prediction_ct_plus_1 = torch.clamp(predicted_cls, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates Sm = log(1+Sm_cls) + weight * log(1+Sm_task)."""
        if true_task_output is None:
            raise ValueError("TaskHead requires true_task_output for surprise calculation.")

        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None or self.last_task_prediction is None:
             self.last_log_surprise_sm = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device)
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             self.last_raw_surprise_sm_task = torch.tensor(float('nan'), device=self.device)
             return self.last_log_surprise_sm

        # --- Calculate RAW Sm_cls ---
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            sm_cls_raw = torch.tensor(1e6, device=self.device)
        else:
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach()

        # --- Calculate RAW Sm_task ---
        if not torch.all(torch.isfinite(self.last_task_prediction)):
            sm_task_raw = torch.tensor(1e6, device=self.device)
        else:
            true_task_output = true_task_output.to(self.device).to(self.last_task_prediction.dtype)
            sm_task_raw = F.mse_loss(self.last_task_prediction, true_task_output.detach(), reduction='mean')
        self.last_raw_surprise_sm_task = sm_task_raw.detach()

        # --- Apply Log Transform ---
        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)
        sm_task_log = torch.log1p(sm_task_raw + EPSILON)

        self.last_log_surprise_sm_cls = sm_cls_log.detach()
        self.last_log_surprise_sm_task = sm_task_log.detach()

        # --- Combine LOG-Surprises ---
        combined_log_surprise = sm_cls_log + self.task_loss_weight * sm_task_log

        if not math.isfinite(combined_log_surprise.item()):
            self.last_log_surprise_sm = self.sm_log_baseline.detach()
        else:
            self.last_log_surprise_sm = combined_log_surprise
            with torch.no_grad():
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * self.last_log_surprise_sm

        return self.last_log_surprise_sm

    def get_task_prediction(self):
        return self.last_task_prediction


# --- Meta-Model (Revised for Weight Decay) ---
class MetaModel(nn.Module):
    """
    Implements DPLD Meta-Model for stability regulation.
    Uses smoothed inputs (EMA Gt, EMA lambda_max) and clips reward.
    Includes weight decay.
    """
    def __init__(self, meta_input_dim, meta_hidden_dim, learning_rate, cls_dim,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1,
                 weight_decay=0.0, # Added weight decay
                 gru_layers=1, instability_weight=1.0, surprise_weight=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.cls_dim = cls_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.stability_target = stability_target
        self.instability_weight = instability_weight
        self.surprise_weight = surprise_weight # Weight for Gt_next in reward

        self.gru = nn.GRU(meta_input_dim, meta_hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.output_layer = nn.Linear(meta_hidden_dim, 1).to(device) # Outputs signal for gamma

        self.optimizer = optim.Adam(list(self.gru.parameters()) + list(self.output_layer.parameters()), lr=learning_rate, weight_decay=weight_decay) # Added weight_decay

        self.last_gru_hidden_state = None
        self.last_meta_input_smoothed = None
        self.last_gamma_offset_signal = None

    def compute_regulation(self, smoothed_gt, smoothed_lambda_max):
        # Handle None/NaN inputs gracefully
        lambda_val = smoothed_lambda_max if smoothed_lambda_max is not None and math.isfinite(smoothed_lambda_max) else self.stability_target
        gt_val = smoothed_gt if smoothed_gt is not None and math.isfinite(smoothed_gt) else 0.0

        meta_input_base = torch.tensor([gt_val, lambda_val], dtype=torch.float32, device=self.device)
        meta_input = meta_input_base.unsqueeze(0).unsqueeze(0)
        self.last_meta_input_smoothed = meta_input.detach().clone()

        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        if hidden_state_input is not None and hidden_state_input.shape[1] != meta_input.shape[0]:
             hidden_state_input = None

        gru_output, self.last_gru_hidden_state = self.gru(meta_input, hidden_state_input)
        last_hidden = gru_output[:, -1, :]
        gamma_offset_signal = self.output_layer(last_hidden).squeeze()
        self.last_gamma_offset_signal = gamma_offset_signal.detach().clone()

        gamma_offset_tanh = torch.tanh(gamma_offset_signal)
        gamma_center = (self.gamma_max + self.gamma_min) / 2.0
        gamma_range = (self.gamma_max - self.gamma_min) / 2.0
        gamma_t = gamma_center + gamma_range * gamma_offset_tanh
        gamma_t = torch.clamp(gamma_t, self.gamma_min, self.gamma_max)

        mmod_t = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                         torch.empty((0,), dtype=torch.float32, device=self.device),
                                         (self.cls_dim,))
        return gamma_t.item(), mmod_t

    def learn(self, Gt_log_next, lambda_max_next):
        """Updates Meta-Model parameters using clipped reward based on log-surprise."""
        grad_norm = 0.0
        if self.last_meta_input_smoothed is None or self.last_gamma_offset_signal is None:
            return 0.0, grad_norm

        if Gt_log_next is None or not math.isfinite(Gt_log_next): return 0.0, grad_norm
        if lambda_max_next is None or not math.isfinite(lambda_max_next): lambda_max_next = self.stability_target

        # --- Calculate Reward (Minimize log-surprise and instability) ---
        surprise_term = self.surprise_weight * torch.tensor(Gt_log_next, device=self.device)
        instability_term = self.instability_weight * F.relu(torch.tensor(lambda_max_next - self.stability_target, device=self.device))
        raw_reward = -(surprise_term + instability_term).detach()
        clipped_reward = torch.clamp(raw_reward, -META_REWARD_CLIP, META_REWARD_CLIP)

        # --- Recompute action prediction ---
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        gru_output_rerun, _ = self.gru(self.last_meta_input_smoothed, hidden_state_input)
        last_hidden_rerun = gru_output_rerun[:, -1, :]
        gamma_offset_signal_mean = self.output_layer(last_hidden_rerun).squeeze()

        meta_action_std = torch.tensor(0.1, device=self.device) # Fixed std for meta-action sampling
        try:
            meta_action_dist = Normal(gamma_offset_signal_mean, meta_action_std)
            log_prob_meta_action = meta_action_dist.log_prob(self.last_gamma_offset_signal)
        except ValueError as e:
             print(f"ERROR creating Meta Normal distribution: {e}")
             self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
             return 0.0, grad_norm

        # --- Policy Gradient Loss ---
        if not torch.isfinite(log_prob_meta_action):
             self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
             return 0.0, grad_norm

        policy_gradient_loss = -clipped_reward * log_prob_meta_action

        if not torch.isfinite(policy_gradient_loss):
            self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
            return 0.0, grad_norm

        # --- Backpropagation ---
        self.optimizer.zero_grad()
        policy_gradient_loss.backward()

        params_to_clip = list(self.gru.parameters()) + list(self.output_layer.parameters())
        total_norm_sq = 0.0
        for p in params_to_clip:
            if p.grad is not None:
                param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
        grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

        nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        self.last_meta_input_smoothed = None
        self.last_gamma_offset_signal = None

        return policy_gradient_loss.item(), grad_norm


# --- DPLD System (Revised for Approx Counterfactual DR) ---
class DPLDSystem(nn.Module):
    """
    Main DPLD system coordinating CLS, Modules, Meta-Model, and Task components.
    Uses LOG-SURPRISE for internal calculations.
    Uses approximate counterfactual difference reward (Option B).
    Includes EMA smoothing for Meta-Model inputs and weight decay.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, noise_std_dev_schedule,
                 env, embedding_dim,
                 entropy_coeff=0.01,
                 ema_alpha=0.99,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1,
                 action_std=DEFAULT_ACTION_STD, task_loss_weight=1.0,
                 weight_decay=0.0, # Added weight decay
                 meta_input_dim=2, clip_cls_norm=True, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules # Number of generic modules
        self.k_sparse_write = k_sparse_write
        self.noise_std_dev_schedule = noise_std_dev_schedule
        self.clip_cls_norm = clip_cls_norm
        self.device = device
        self.env = env
        self.ema_alpha = ema_alpha

        self.ct = self._init_cls()

        num_vocab_size = self.env.get_vocab_size_numbers()
        op_vocab_size = self.env.get_num_ops()
        self.math_encoder = MathEncoder(num_vocab_size, op_vocab_size, embedding_dim, cls_dim, k_sparse_write, device).to(device)
        self.task_head = TaskHead(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                                  entropy_coeff=entropy_coeff,
                                  action_std=action_std, task_loss_weight=task_loss_weight,
                                  weight_decay=weight_decay, device=device).to(device)

        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                             entropy_coeff=entropy_coeff,
                             action_std=action_std, task_loss_weight=0.0, # Generic modules have 0 task weight
                             weight_decay=weight_decay, device=device)
            for _ in range(num_modules)
        ])

        self.meta_model = MetaModel(meta_input_dim, meta_hidden_dim, meta_lr, cls_dim,
                                    gamma_min, gamma_max, stability_target,
                                    weight_decay=weight_decay, device=device).to(device)

        self.gt_log_ema = None # EMA for LOG global surprise
        self.lambda_max_ema = None
        self.last_log_surprises_sm = None # Store previous step's surprises for DR Option B

    def _init_cls(self):
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.01
        sparse_ct = sparsify_vector(initial_dense, self.k_sparse_write).to_sparse_coo()
        return sparse_ct.coalesce()

    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation (Unchanged)."""
        ct = ct if ct.is_sparse else ct.to_sparse_coo()
        sum_im = sum_im if sum_im.is_sparse else sum_im.to_sparse_coo()
        mmodt = mmodt if mmodt.is_sparse else mmodt.to_sparse_coo()

        if not torch.all(torch.isfinite(ct.values())):
             ct = self._init_cls()
        if not torch.all(torch.isfinite(sum_im.values())):
             sum_im = self._init_cls()
        if not torch.all(torch.isfinite(mmodt.values())):
             mmodt = self._init_cls()

        decayed_ct = (1.0 - gamma_t) * ct
        combined_inputs = (decayed_ct + sum_im + mmodt).coalesce()
        noise_et = torch.randn(self.cls_dim, device=self.device) * noise_std_dev
        ct_plus_1_dense = combined_inputs.to_dense() + noise_et

        if not torch.all(torch.isfinite(ct_plus_1_dense)):
            ct_plus_1_dense = torch.nan_to_num(ct_plus_1_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)

        if self.clip_cls_norm:
            norm = torch.linalg.norm(ct_plus_1_dense)
            if norm > CLS_NORM_CLIP_VAL:
                 ct_plus_1_dense = ct_plus_1_dense * (CLS_NORM_CLIP_VAL / (norm + EPSILON))

        ct_plus_1_sparse_vals = sparsify_vector(ct_plus_1_dense, self.k_sparse_write)
        sparse_indices = torch.where(ct_plus_1_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = ct_plus_1_sparse_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             ct_plus_1 = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=ct_plus_1_dense.dtype)
        else:
             ct_plus_1 = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=ct_plus_1_dense.dtype, device=self.device),
                                                 (self.cls_dim,))
        return ct_plus_1.coalesce()


    def get_dynamics_map(self, fixed_gamma, fixed_noise_std=0.0):
         """Returns a function for LE estimation (Unchanged from previous revision)."""
         gamma_val = float(fixed_gamma)
         generic_module_params = []
         for module in self.pred_modules:
              generic_module_params.append({
                  'fm_params': [p.detach().clone() for p in module.fm.parameters()],
                  'qm': module.qm.detach().clone(), 'sm_baseline': module.sm_log_baseline.detach().clone(),
                  'k_sparse': module.k_sparse_write, 'surprise_scale': module.surprise_scale_factor,
                  'action_std': module.action_std # Include action_std if needed for dynamics map
              })
         task_head_params = {
             'cls_pred_params': [p.detach().clone() for p in self.task_head.cls_pred_layer.parameters()],
             'qm': self.task_head.qm.detach().clone(), 'sm_baseline': self.task_head.sm_log_baseline.detach().clone(),
             'k_sparse': self.task_head.k_sparse_write, 'surprise_scale': self.task_head.surprise_scale_factor,
             'hidden_dim': self.task_head.module_hidden_dim, 'cls_dim': self.cls_dim,
             'action_std': self.task_head.action_std # Include action_std
         }

         def dynamics_map(state_t_dense):
             if not torch.all(torch.isfinite(state_t_dense)): return state_t_dense.detach()
             state_t_sparse = state_t_dense.detach().to_sparse_coo()
             sum_im = self._init_cls()
             with torch.no_grad():
                 for params in generic_module_params:
                      x = state_t_sparse.to_dense()
                      try:
                          for i in range(0, len(params['fm_params']), 2):
                               weight = params['fm_params'][i]; bias = params['fm_params'][i+1] if i+1 < len(params['fm_params']) else None
                               x = F.linear(x, weight, bias)
                               if i+2 < len(params['fm_params']): x = F.relu(x)
                          pred_cls = torch.clamp(x, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
                      except Exception: pred_cls = torch.zeros(self.cls_dim, device=self.device)
                      vm = pred_cls; gate_raw_score = params['qm'] * state_t_sparse.to_dense(); gate_activation_gm = torch.sigmoid(gate_raw_score / 1.0)
                      log_surprise_diff = 0.0 # Hardcoded 0 for LE estimation map
                      influence_scalar_am = 1.0 + 1.0 * torch.tanh(torch.tensor(params['surprise_scale'] * log_surprise_diff, device=self.device))
                      influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                      intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                      intermediate_write_vector = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)
                      # Use mean action for LE dynamics map? Or sample? Using mean for now.
                      write_vector_im_sparse_vals = sparsify_vector(intermediate_write_vector, params['k_sparse'])
                      sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0); sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                      if sparse_indices.numel() > 0: sum_im += torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=intermediate_write_vector.dtype)

                 params = task_head_params
                 try: # Simplified TaskHead dynamics for LE
                     temp_hidden = F.relu(F.linear(state_t_sparse.to_dense(), torch.zeros(params['hidden_dim'], params['cls_dim'], device=self.device)))
                     pred_cls = F.linear(temp_hidden, params['cls_pred_params'][0], params['cls_pred_params'][1])
                     pred_cls = torch.clamp(pred_cls, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
                 except Exception: pred_cls = torch.zeros(self.cls_dim, device=self.device)
                 vm = pred_cls; gate_raw_score = params['qm'] * state_t_sparse.to_dense(); gate_activation_gm = torch.sigmoid(gate_raw_score / 1.0)
                 log_surprise_diff = 0.0
                 influence_scalar_am = 1.0 + 1.0 * torch.tanh(torch.tensor(params['surprise_scale'] * log_surprise_diff, device=self.device))
                 influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                 intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                 intermediate_write_vector = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)
                 # Use mean action for LE dynamics map
                 write_vector_im_sparse_vals = sparsify_vector(intermediate_write_vector, params['k_sparse'])
                 sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0); sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                 if sparse_indices.numel() > 0: sum_im += torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=intermediate_write_vector.dtype)

                 gamma_t = gamma_val; mmodt = self._init_cls()
                 next_state_sparse = self.cls_update_rule(state_t_sparse, sum_im.coalesce(), mmodt, gamma_t, fixed_noise_std)
                 next_state_dense = next_state_sparse.to_dense()
                 if not torch.all(torch.isfinite(next_state_dense)):
                     next_state_dense = torch.nan_to_num(next_state_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)
             return next_state_dense
         return dynamics_map

    def step(self, current_step_num, task_input, estimate_le=False):
        """Performs one full step using LOG-SURPRISE and Approx Counterfactual DR."""

        a, op_idx, b, true_answer_c = task_input

        if not torch.all(torch.isfinite(self.ct.values())):
             self.ct = self._init_cls()
             self.gt_log_ema = None
             self.lambda_max_ema = None
             self.last_log_surprises_sm = None # Reset stored surprises
             for module in self.pred_modules: module.sm_log_baseline.fill_(0.0)
             self.task_head.sm_log_baseline.fill_(0.0)
             self.task_head.last_gru_hidden_state = None
             self.meta_model.last_gru_hidden_state = None

        ct_prev_dense = self.ct.to_dense().detach()

        # --- 1. Module Predictions ---
        all_modules = self.pred_modules + [self.task_head]
        for module in all_modules:
            _ = module.predict(self.ct)

        # --- 2. Meta-Model Regulation ---
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.gt_log_ema, self.lambda_max_ema)

        # --- 3. Module Write Vectors ---
        sum_im = self._init_cls()
        i_math = self.math_encoder(a, op_idx, b)
        sum_im += i_math
        for module in all_modules:
             im = module.generate_write_vector(self.ct)
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual LOG-Surprises & Global LOG-Surprise ---
        current_log_surprises_sm = []
        log_surprises_sm_cls = []
        log_surprises_sm_task = []
        raw_surprises_sm_cls = []
        raw_surprises_sm_task = []
        global_log_surprise_gt_sum = 0.0
        valid_surprises = 0
        for module in self.pred_modules:
             sm_log = module.calculate_surprise(ct_plus_1)
             current_log_surprises_sm.append(sm_log.detach()) # Store current step's surprises
             log_surprises_sm_cls.append(module.last_log_surprise_sm_cls)
             log_surprises_sm_task.append(module.last_log_surprise_sm_task)
             raw_surprises_sm_cls.append(module.last_raw_surprise_sm_cls)
             raw_surprises_sm_task.append(module.last_raw_surprise_sm_task)
             if math.isfinite(sm_log.item()): global_log_surprise_gt_sum += sm_log; valid_surprises += 1
        sm_log_task_head = self.task_head.calculate_surprise(ct_plus_1, true_answer_c)
        current_log_surprises_sm.append(sm_log_task_head.detach()) # Store current step's surprises
        log_surprises_sm_cls.append(self.task_head.last_log_surprise_sm_cls)
        log_surprises_sm_task.append(self.task_head.last_log_surprise_sm_task)
        raw_surprises_sm_cls.append(self.task_head.last_raw_surprise_sm_cls)
        raw_surprises_sm_task.append(self.task_head.last_raw_surprise_sm_task)
        if math.isfinite(sm_log_task_head.item()): global_log_surprise_gt_sum += sm_log_task_head; valid_surprises += 1

        global_log_surprise_gt = (global_log_surprise_gt_sum / valid_surprises) if valid_surprises > 0 else torch.tensor(float('nan'), device=self.device)

        # --- 6. Calculate Approx Counterfactual Difference Rewards (Option B) & Normalize ---
        raw_rewards_rm_log = []
        if self.last_log_surprises_sm is not None and len(self.last_log_surprises_sm) == len(all_modules):
            num_prev_modules = len(self.last_log_surprises_sm)
            for i in range(num_prev_modules):
                # Calculate baseline: average of *other* modules' *previous* log-surprise
                other_prev_surprises = []
                for j in range(num_prev_modules):
                    if i == j: continue
                    prev_sm_log = self.last_log_surprises_sm[j]
                    if prev_sm_log is not None and math.isfinite(prev_sm_log.item()):
                         other_prev_surprises.append(prev_sm_log.item())

                baseline_m_prev = np.mean(other_prev_surprises) if other_prev_surprises else 0.0
                current_gt_log_item = global_log_surprise_gt.item()

                if math.isfinite(baseline_m_prev) and math.isfinite(current_gt_log_item):
                    rm_log = baseline_m_prev - current_gt_log_item
                    raw_rewards_rm_log.append(rm_log)
                else:
                    raw_rewards_rm_log.append(0.0) # Default reward if calculation fails
        else:
            # Handle first step or mismatch
            raw_rewards_rm_log = [0.0] * len(all_modules)

        normalized_rewards_rm = []
        mean_rm_log_val = 0.0
        std_rm_log_val = 0.0
        if len(raw_rewards_rm_log) > 0:
             rewards_tensor = torch.tensor(raw_rewards_rm_log, device=self.device)
             mean_rm_log = rewards_tensor.mean()
             std_rm_log = rewards_tensor.std()
             mean_rm_log_val = mean_rm_log.item()
             std_rm_log_val = std_rm_log.item()
             if std_rm_log > EPSILON:
                 normalized_rewards_rm = ((rewards_tensor - mean_rm_log) / std_rm_log).tolist()
             else:
                 normalized_rewards_rm = torch.zeros_like(rewards_tensor).tolist()
        else:
             normalized_rewards_rm = [0.0] * len(all_modules)

        # --- Store current surprises for next step's DR calculation ---
        self.last_log_surprises_sm = [s.detach().clone() for s in current_log_surprises_sm] # Store detached clones

        # --- 7. Estimate LE (Optional) ---
        lambda_max_estimate = None
        if estimate_le and torch.all(torch.isfinite(ct_prev_dense)):
            dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
            lambda_max_estimate = estimate_lyapunov_exponent(dynamics_map_for_le, ct_prev_dense, device=self.device)
            if lambda_max_estimate is not None and not math.isfinite(lambda_max_estimate):
                 lambda_max_estimate = None

        # --- 8. Update Meta-Model EMA Inputs ---
        current_gt_log_val = global_log_surprise_gt.item()
        if math.isfinite(current_gt_log_val):
            if self.gt_log_ema is None: self.gt_log_ema = current_gt_log_val
            else: self.gt_log_ema = (1 - self.ema_alpha) * current_gt_log_val + self.ema_alpha * self.gt_log_ema

        current_lambda_val = lambda_max_estimate
        if current_lambda_val is not None and math.isfinite(current_lambda_val):
             if self.lambda_max_ema is None: self.lambda_max_ema = current_lambda_val
             else: self.lambda_max_ema = (1 - self.ema_alpha) * current_lambda_val + self.ema_alpha * self.lambda_max_ema

        # --- 9. Trigger Meta-Model Learning ---
        meta_loss_item, meta_grad_norm = self.meta_model.learn(self.gt_log_ema, self.lambda_max_ema)

        # --- 10. Trigger Module Learning ---
        module_policy_losses = []
        module_task_losses_log = []
        module_entropies = []
        module_grad_norms = []
        for i, module in enumerate(all_modules):
            norm_reward = normalized_rewards_rm[i] if i < len(normalized_rewards_rm) else 0.0
            policy_loss, task_loss_log, entropy, grad_norm = module.learn(norm_reward)
            module_policy_losses.append(policy_loss)
            module_task_losses_log.append(task_loss_log)
            module_entropies.append(entropy)
            module_grad_norms.append(grad_norm)

        # --- 11. Update State ---
        self.ct = ct_plus_1

        # --- 12. Calculate Task Accuracy ---
        task_accuracy = 0.0
        task_pred_val = self.task_head.get_task_prediction()
        if task_pred_val is not None and math.isfinite(task_pred_val.item()):
             task_accuracy = 1.0 if torch.abs(task_pred_val - true_answer_c) < 0.5 else 0.0

        # --- 13. Return Metrics (Log-Surprise Based, Approx Counterfactual DR) ---
        finite_sm_log = [s.item() for s in current_log_surprises_sm if s is not None and math.isfinite(s.item())] # Use current step's surprises
        finite_sm_log_cls = [s.item() for s in log_surprises_sm_cls if s is not None and math.isfinite(s.item())]
        finite_sm_log_task = [s.item() for s in log_surprises_sm_task if s is not None and math.isfinite(s.item())]
        finite_sm_raw_cls = [s.item() for s in raw_surprises_sm_cls if s is not None and math.isfinite(s.item())]
        finite_sm_raw_task = [s.item() for s in raw_surprises_sm_task if s is not None and math.isfinite(s.item())]
        finite_norm_rewards = [r for r in normalized_rewards_rm if math.isfinite(r)]
        finite_policy_losses = [l for l in module_policy_losses if l is not None and math.isfinite(l)]
        finite_entropies = [e for e in module_entropies if e is not None and math.isfinite(e)]
        finite_module_grads = [g for g in module_grad_norms if g is not None and math.isfinite(g)]

        metrics = {
            # Log-Surprise Metrics
            "Gt_log": global_log_surprise_gt.item() if math.isfinite(global_log_surprise_gt.item()) else float('nan'),
            "Gt_log_EMA": self.gt_log_ema,
            "Sm_log_avg": np.mean(finite_sm_log) if finite_sm_log else float('nan'),
            "Sm_log_std": np.std(finite_sm_log) if len(finite_sm_log) > 1 else 0.0,
            "Sm_log_cls_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_log_task_avg": np.mean(finite_sm_log_task) if finite_sm_log_task else float('nan'),
            "TaskHead_Sm_log_task": finite_sm_log_task[-1] if finite_sm_log_task else float('nan'),
            # Raw Surprise Metrics (for scale reference)
            "Sm_raw_cls_avg": np.mean(finite_sm_raw_cls) if finite_sm_raw_cls else float('nan'),
            "Sm_raw_task_avg": np.mean(finite_sm_raw_task) if finite_sm_raw_task else float('nan'),
            "TaskHead_Sm_raw_task": finite_sm_raw_task[-1] if finite_sm_raw_task else float('nan'),
            # Task Accuracy
            "TaskAccuracy": task_accuracy,
            # Reward Metrics (Approx Counterfactual DR)
            "Rm_log_raw_avg": mean_rm_log_val if math.isfinite(mean_rm_log_val) else float('nan'), # Now Rm_log = baseline_prev - Gt_curr
            "Rm_log_raw_std": std_rm_log_val if math.isfinite(std_rm_log_val) else float('nan'),
            "Rm_norm_avg": np.mean(finite_norm_rewards) if finite_norm_rewards else float('nan'),
            "Rm_norm_std": np.std(finite_norm_rewards) if len(finite_norm_rewards) > 1 else 0.0,
            # Stability & Dynamics
            "lambda_max_est": lambda_max_estimate,
            "lambda_max_EMA": self.lambda_max_ema,
            "gamma_t": gamma_t,
            "noise_std": noise_std_dev,
            # Learning Performance
            "module_loss_avg": np.mean(finite_policy_losses) if finite_policy_losses else float('nan'),
            "meta_loss": meta_loss_item if math.isfinite(meta_loss_item) else float('nan'),
            "module_entropy_avg": np.mean(finite_entropies) if finite_entropies else float('nan'),
            "module_grad_norm_avg": np.mean(finite_module_grads) if finite_module_grads else float('nan'),
            "meta_grad_norm": meta_grad_norm if math.isfinite(meta_grad_norm) else float('nan'),
            # CLS State
            "cls_norm": torch.linalg.norm(self.ct.to_dense()).item() if torch.all(torch.isfinite(self.ct.values())) else float('nan'),
            "cls_density": self.ct._nnz() / self.cls_dim if self.ct._nnz() is not None and self.cls_dim > 0 else float('nan')
        }
        return metrics