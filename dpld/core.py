# dpld/core.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math # For checking nan/inf

from utils import estimate_lyapunov_exponent, sparsify_vector # Assuming these are in utils.py

# --- Constants ---
EPSILON = 1e-8
DEFAULT_ACTION_STD = 0.5
CLS_NORM_CLIP_VAL = 100.0
MODULE_OUTPUT_CLIP_VAL = 100.0
ACTION_MEAN_CLIP_VAL = 5.0
GRAD_CLIP_NORM = 1.0
META_REWARD_CLIP = 10.0
DEFAULT_PREDICTION_LOSS_WEIGHT = 1.0


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


# --- Predictive Module (Base Class - Combined Loss) ---
class PredictiveModule(nn.Module):
    """
    Base class for DPLD modules: Read, Predict, Write.
    Uses combined loss: Policy Loss (for qm) + Prediction Loss (for fm).
    Uses simple reward = -local_log_surprise_sm for policy loss.
    Uses LOG-SURPRISE internally and for influence scaling.
    Includes stability clamping, action mean clipping, and weight decay.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0,
                 surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT,
                 task_loss_weight=0.0, # Task weight is 0
                 weight_decay=1e-5, # Default weight decay
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
        self.prediction_loss_weight = prediction_loss_weight
        self.task_loss_weight = task_loss_weight # Will be 0 for base modules

        # --- fm: Prediction Network (requires grad for prediction loss) ---
        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)

        # --- qm: Gating Parameter (requires grad for policy loss) ---
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01)

        # --- Initialize Buffers and State ---
        self.register_buffer('sm_log_baseline', torch.tensor(0.0, device=device))
        self.last_prediction_ct_plus_1 = None
        self.last_log_surprise_sm = None
        # --- MODIFICATION: Initialize last_log_surprise_sm_cls ---
        # Initialize with a detached tensor value (e.g., initial baseline)
        self.last_log_surprise_sm_cls = torch.tensor(0.0, device=device)
        # --- END MODIFICATION ---
        self.last_log_surprise_sm_task = torch.tensor(0.0, device=device) # Initialized as tensor
        self.last_raw_surprise_sm_cls = None
        self.last_raw_surprise_sm_task = None
        self.last_log_prob = None
        self.last_action_dist = None

        # Optimizer includes fm and qm parameters
        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate, weight_decay=weight_decay)

    def predict(self, ct):
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        prediction = self.fm(ct_dense.detach()) # Detach input ct, but keep grad path through fm
        prediction = torch.clamp(prediction, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_prediction_ct_plus_1 = prediction # Store prediction with grad history
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates local LOG-SURPRISE Sm (only CLS part for base module)."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             # Use initial baseline value if no prediction yet
             self.last_log_surprise_sm = torch.tensor(0.0, device=self.device) # No grad history
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device)
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             self.last_raw_surprise_sm_task = torch.tensor(0.0, device=self.device)
             return self.last_log_surprise_sm # Return detached value

        # --- Calculate RAW Sm_cls ---
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            sm_cls_raw = torch.mean((self.last_prediction_ct_plus_1 - actual_ct_plus_1_dense.detach())**2) \
                         if torch.is_grad_enabled() else torch.tensor(1e6, device=self.device)
            if not torch.isfinite(sm_cls_raw):
                 sm_cls_raw = torch.tensor(1e6, device=self.device)
        else:
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach() # Store detached raw value

        sm_task_raw = torch.tensor(0.0, device=self.device)
        self.last_raw_surprise_sm_task = sm_task_raw.detach()

        # --- Calculate LOG Sm_cls ---
        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)
        sm_task_log = torch.tensor(0.0, device=self.device) # No task component

        # Store detached versions for baseline update and logging
        self.last_log_surprise_sm_cls = sm_cls_log.detach()
        self.last_log_surprise_sm_task = sm_task_log.detach()

        # Store combined log surprise WITH grad history
        combined_log_surprise = sm_cls_log # Task weight is 0

        if not math.isfinite(combined_log_surprise.item()):
             self.last_log_surprise_sm = torch.tensor(self.sm_log_baseline.item() + 1.0, device=self.device) # No grad history
        else:
            self.last_log_surprise_sm = combined_log_surprise
            with torch.no_grad():
                # Ensure baseline update uses a finite value
                update_val = self.last_log_surprise_sm_cls if torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * update_val

        # Return the grad-enabled log surprise for use in prediction loss
        return self.last_log_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse write vector Im using LOG-SURPRISE for alpha_m."""
        if self.last_prediction_ct_plus_1 is None:
             # Handle case where predict hasn't run (shouldn't happen after step 0)
             # For action mean calculation, we need vm. Let's use zeros if no prediction.
             vm = torch.zeros(self.cls_dim, device=self.device)
             current_log_surprise_detached = self.sm_log_baseline.detach() # Use baseline if no surprise calculated
        else:
             # Use DETACHED log surprise for influence scaling
             # Check if last_log_surprise_sm_cls is valid, otherwise use baseline
             current_log_surprise_detached = self.last_log_surprise_sm_cls if self.last_log_surprise_sm_cls is not None and torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline.detach()
             # vm is the prediction from fm, DETACH it for action mean calculation
             vm = self.last_prediction_ct_plus_1.detach()

        log_surprise_diff = current_log_surprise_detached - self.sm_log_baseline.detach()

        ct_dense = ct.to_dense() if ct.is_sparse else ct

        # --- Gating (Requires Grad through qm) ---
        tau_g = 1.0
        gate_raw_score = self.qm * ct_dense.detach()
        gate_activation_gm = torch.sigmoid(gate_raw_score / tau_g) # Has grad path to qm

        # --- Influence (No Grad) ---
        alpha_base = 1.0; alpha_scale = 1.0
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * log_surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0).detach() # Detach influence

        # --- Action Mean Calculation (Requires Grad through qm ONLY) ---
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm) # Grad path only to qm
        action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL) # Grad path only to qm

        if not torch.all(torch.isfinite(action_mean)):
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=ACTION_MEAN_CLIP_VAL, neginf=-ACTION_MEAN_CLIP_VAL)

        action_std_tensor = torch.full_like(action_mean, fill_value=self.action_std)

        try:
            dist = Normal(action_mean, action_std_tensor)
            self.last_action_dist = dist # Store for entropy calculation
            dense_write_vector_sampled = dist.sample()
            self.last_log_prob = dist.log_prob(dense_write_vector_sampled.detach()).sum()
        except ValueError as e:
             print(f"ERROR creating Normal distribution in {self.__class__.__name__}: {e}")
             dense_write_vector_sampled = torch.zeros_like(action_mean)
             self.last_log_prob = None
             self.last_action_dist = None

        # Sparsify the sampled action
        write_vector_im_sparse_vals = sparsify_vector(dense_write_vector_sampled.detach(), self.k_sparse_write)
        sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=dense_write_vector_sampled.dtype)
        else:
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=dense_write_vector_sampled.dtype, device=self.device),
                                                 (self.cls_dim,))
        return im_sparse.coalesce()

    def learn(self, local_log_surprise_sm_detached):
        """
        Updates module parameters (fm and qm) using combined loss.
        Args:
            local_log_surprise_sm_detached: The detached log-surprise for reward calculation.
        Returns:
            total_loss_item, policy_loss_item, prediction_loss_item, entropy_item, grad_norm
        """
        policy_loss_item = 0.0
        prediction_loss_item = 0.0
        entropy_item = 0.0
        grad_norm = 0.0

        # --- 1. Calculate Policy Loss Component (for qm) ---
        policy_loss = torch.tensor(0.0, device=self.device)
        if self.last_log_prob is not None and self.last_action_dist is not None and \
           local_log_surprise_sm_detached is not None and math.isfinite(local_log_surprise_sm_detached.item()):

            reward_tensor = -local_log_surprise_sm_detached # Simple reward

            entropy_tensor = torch.tensor(0.0, device=self.device)
            try:
                entropy_tensor = self.last_action_dist.entropy().sum()
                if not torch.isfinite(entropy_tensor): entropy_tensor = torch.tensor(0.0, device=self.device)
                entropy_item = entropy_tensor.item()
            except Exception: entropy_tensor = torch.tensor(0.0, device=self.device); entropy_item = 0.0

            entropy_term = self.entropy_coeff * entropy_tensor

            if torch.isfinite(self.last_log_prob):
                policy_loss = -(reward_tensor * self.last_log_prob + entropy_term)
                if not torch.isfinite(policy_loss):
                    policy_loss = torch.tensor(0.0, device=self.device)
            else:
                policy_loss = torch.tensor(0.0, device=self.device)
        policy_loss_item = policy_loss.item() # Store item before combining

        # --- 2. Calculate Prediction Loss Component (for fm) ---
        prediction_loss = torch.tensor(0.0, device=self.device)
        # self.last_log_surprise_sm has grad history back to fm (set in calculate_surprise)
        if self.last_log_surprise_sm is not None and torch.is_tensor(self.last_log_surprise_sm) and self.last_log_surprise_sm.requires_grad:
            if torch.isfinite(self.last_log_surprise_sm):
                prediction_loss = self.prediction_loss_weight * self.last_log_surprise_sm
            else:
                prediction_loss = torch.tensor(0.0, device=self.device) # Zero out if NaN/Inf
        prediction_loss_item = prediction_loss.item() # Store item before combining

        # --- 3. Combine Losses ---
        total_loss = policy_loss + prediction_loss

        # --- 4. Backpropagation and Optimizer Step ---
        if torch.isfinite(total_loss) and total_loss.requires_grad: # Check requires_grad
            self.optimizer.zero_grad()
            total_loss.backward() # Calculate gradients for fm and qm

            params_to_clip = list(self.fm.parameters()) + [self.qm]

            total_norm_sq = 0.0
            for p in params_to_clip:
                if p.grad is not None:
                     param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                     if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
            grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

            nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
            self.optimizer.step()
        else:
            # If loss is invalid or no grads, skip optimizer step
            grad_norm = 0.0
            # If loss is NaN/Inf, print a warning
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN/Inf total_loss in {self.__class__.__name__}.learn(). Policy: {policy_loss_item}, Pred: {prediction_loss_item}")


        # Reset state for next step
        self.last_log_prob = None
        self.last_action_dist = None
        # Keep self.last_log_surprise_sm as it's set by calculate_surprise next step

        return total_loss.item(), policy_loss_item, prediction_loss_item, entropy_item, grad_norm


# --- Task Head Module (Combined Loss - Inherits changes) ---
class TaskHead(PredictiveModule):
    """
    Specialized PredictiveModule for the arithmetic task.
    Uses combined loss: Policy Loss (for qm) + Prediction Loss (for GRU/Linear).
    Uses simple reward = -local_log_surprise_sm (which is only CLS surprise).
    Uses LOG-SURPRISE. Internal model `fm` uses a GRU.
    TASK SURPRISE CALCULATION REMAINS DISABLED.
    Includes weight decay.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT, # Added
                 task_loss_weight=0.0, # TASK WEIGHT DISABLED
                 weight_decay=1e-5, # Default weight decay
                 gru_layers=1, device='cpu'):
        # Initialize PredictiveModule with task_loss_weight=0.0 and prediction_loss_weight
        super().__init__(cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                         entropy_coeff, surprise_scale_factor, surprise_baseline_ema, action_std,
                         prediction_loss_weight=prediction_loss_weight, # Pass prediction weight
                         task_loss_weight=0.0, # Force task weight to 0
                         weight_decay=weight_decay, device=device) # Pass weight_decay

        # --- fm: Prediction Network (GRU + Linear layers) ---
        # Redefine fm components for TaskHead
        self.gru = nn.GRU(cls_dim, module_hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.task_out_layer = nn.Linear(module_hidden_dim, 1).to(device) # Predicts task answer
        self.cls_pred_layer = nn.Linear(module_hidden_dim, cls_dim).to(device) # Predicts next CLS state

        # Optimizer includes GRU, output layers, and qm
        # Re-define optimizer to include TaskHead specific layers instead of base fm
        self.optimizer = optim.Adam(
            list(self.gru.parameters()) +
            list(self.task_out_layer.parameters()) +
            list(self.cls_pred_layer.parameters()) +
            [self.qm], # qm is inherited
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.last_gru_hidden_state = None
        self.last_task_prediction = None # Stores task prediction (no grad needed after loss calc)

    def predict(self, ct):
        """Predicts both next CLS state and task answer."""
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        ct_input = ct_dense.detach().unsqueeze(0).unsqueeze(0) # Detach input ct
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None

        if hidden_state_input is not None and hidden_state_input.shape[1] != ct_input.shape[0]:
             hidden_state_input = None

        # GRU requires grad for fm update
        gru_output, next_hidden_state = self.gru(ct_input, hidden_state_input)
        # Detach hidden state before storing if not needed for BPTT (which we aren't doing across steps here)
        self.last_gru_hidden_state = next_hidden_state.detach()

        last_hidden = gru_output[:, -1, :] # This depends on GRU params

        # Task prediction (no grad needed after loss calculation)
        predicted_answer = self.task_out_layer(last_hidden).squeeze() # Depends on task_out params
        self.last_task_prediction = torch.clamp(predicted_answer, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL).detach() # Store detached

        # CLS prediction (needs grad history for prediction loss)
        predicted_cls = self.cls_pred_layer(last_hidden).squeeze(0) # Depends on cls_pred params
        self.last_prediction_ct_plus_1 = torch.clamp(predicted_cls, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL) # Store with grad history

        # Return the CLS prediction (with grad history)
        return self.last_prediction_ct_plus_1

    # calculate_surprise is inherited from PredictiveModule (task_loss_weight=0)

    def get_task_prediction(self):
        # Returns the detached task prediction made in the predict step
        return self.last_task_prediction

    # learn method is inherited from PredictiveModule, which now uses combined loss
    # The optimizer defined in __init__ will correctly update GRU, Linear, and qm params.


# --- Meta-Model (Unchanged) ---
class MetaModel(nn.Module):
    """
    Implements DPLD Meta-Model for stability regulation.
    LEARNING IS ENABLED. Uses smoothed inputs (EMA Gt, EMA lambda_max) and clips reward.
    Includes weight decay. Stability target can be set high to ignore LE.
    """
    def __init__(self, meta_input_dim, meta_hidden_dim, learning_rate, cls_dim,
                 gamma_min=0.01, gamma_max=0.2, stability_target=-100.0,
                 weight_decay=1e-5, # Default weight decay
                 gru_layers=1, instability_weight=1.0, surprise_weight=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.cls_dim = cls_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.stability_target = stability_target
        self.instability_weight = instability_weight
        self.surprise_weight = surprise_weight

        self.gru = nn.GRU(meta_input_dim, meta_hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.output_layer = nn.Linear(meta_hidden_dim, 1).to(device)

        self.optimizer = optim.Adam(list(self.gru.parameters()) + list(self.output_layer.parameters()), lr=learning_rate, weight_decay=weight_decay)

        self.last_gru_hidden_state = None
        self.last_meta_input_smoothed = None
        self.last_gamma_offset_signal = None

    def compute_regulation(self, smoothed_gt, smoothed_lambda_max):
        lambda_val = smoothed_lambda_max if smoothed_lambda_max is not None and math.isfinite(smoothed_lambda_max) else self.stability_target
        gt_val = smoothed_gt if smoothed_gt is not None and math.isfinite(smoothed_gt) else 0.0

        meta_input_base = torch.tensor([gt_val, lambda_val], dtype=torch.float32, device=self.device)
        meta_input = meta_input_base.unsqueeze(0).unsqueeze(0)
        self.last_meta_input_smoothed = meta_input.detach().clone()

        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        if hidden_state_input is not None and hidden_state_input.shape[1] != meta_input.shape[0]:
             hidden_state_input = None

        with torch.no_grad():
            gru_output, next_hidden_state = self.gru(meta_input, hidden_state_input)
            self.last_gru_hidden_state = next_hidden_state.detach() # Store detached hidden state
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

        surprise_term = self.surprise_weight * torch.tensor(Gt_log_next, device=self.device)
        instability_term = self.instability_weight * F.relu(torch.tensor(lambda_max_next - self.stability_target, device=self.device))
        raw_reward = -(surprise_term + instability_term).detach()
        clipped_reward = torch.clamp(raw_reward, -META_REWARD_CLIP, META_REWARD_CLIP)

        # Rerun the forward pass to get grad history
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None # Use detached hidden state
        gru_output_rerun, _ = self.gru(self.last_meta_input_smoothed, hidden_state_input)
        last_hidden_rerun = gru_output_rerun[:, -1, :]
        gamma_offset_signal_mean = self.output_layer(last_hidden_rerun).squeeze() # Has grad history

        meta_action_std = torch.tensor(0.1, device=self.device)
        try:
            meta_action_dist = Normal(gamma_offset_signal_mean, meta_action_std)
            log_prob_meta_action = meta_action_dist.log_prob(self.last_gamma_offset_signal) # Use stored action
        except ValueError as e:
             print(f"ERROR creating Meta Normal distribution: {e}")
             self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
             return 0.0, grad_norm

        if not torch.isfinite(log_prob_meta_action):
             self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
             return 0.0, grad_norm

        policy_gradient_loss = -clipped_reward * log_prob_meta_action

        if not torch.isfinite(policy_gradient_loss):
            self.last_meta_input_smoothed = None; self.last_gamma_offset_signal = None
            return 0.0, grad_norm

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


# --- DPLD System (Unchanged from Rev 7 - Relies on module changes) ---
class DPLDSystem(nn.Module):
    """
    Main DPLD system coordinating CLS, Modules, Meta-Model, and Task components.
    Uses combined loss for modules (Policy Loss for qm, Prediction Loss for fm).
    Uses LOG-SURPRISE for internal calculations.
    Includes EMA smoothing, weight decay, action mean clipping.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, noise_std_dev_schedule,
                 env, embedding_dim,
                 entropy_coeff=0.01,
                 ema_alpha=0.99,
                 gamma_min=0.01, gamma_max=0.2, stability_target=-100.0,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT, # Added
                 task_loss_weight=0.0, # Task weight is 0
                 weight_decay=1e-5, # Default weight decay
                 meta_input_dim=2, clip_cls_norm=True, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules
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
                                  action_std=action_std,
                                  prediction_loss_weight=prediction_loss_weight, # Pass weight
                                  task_loss_weight=0.0, # Force 0
                                  weight_decay=weight_decay, device=device).to(device)

        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                             entropy_coeff=entropy_coeff,
                             action_std=action_std,
                             prediction_loss_weight=prediction_loss_weight, # Pass weight
                             task_loss_weight=0.0,
                             weight_decay=weight_decay, device=device)
            for _ in range(num_modules)
        ])

        self.meta_model = MetaModel(meta_input_dim, meta_hidden_dim, meta_lr, cls_dim,
                                    gamma_min, gamma_max, stability_target,
                                    weight_decay=weight_decay, device=device).to(device)

        self.gt_log_ema = None
        self.lambda_max_ema = None

    def _init_cls(self):
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.01
        sparse_ct = sparsify_vector(initial_dense, self.k_sparse_write).to_sparse_coo()
        return sparse_ct.coalesce()

    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation (Unchanged)."""
        ct = ct if ct.is_sparse else ct.to_sparse_coo()
        sum_im = sum_im if sum_im.is_sparse else sum_im.to_sparse_coo()
        mmodt = mmodt if mmodt.is_sparse else mmodt.to_sparse_coo()

        if not torch.all(torch.isfinite(ct.values())): ct = self._init_cls()
        if not torch.all(torch.isfinite(sum_im.values())): sum_im = self._init_cls()
        if not torch.all(torch.isfinite(mmodt.values())): mmodt = self._init_cls()

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
         """Returns a function for LE estimation (Unchanged)."""
         gamma_val = float(fixed_gamma)
         generic_module_params = []
         for module in self.pred_modules:
              generic_module_params.append({
                  'fm_params': [p.detach().clone() for p in module.fm.parameters()],
                  'qm': module.qm.detach().clone(), 'sm_baseline': module.sm_log_baseline.detach().clone(),
                  'k_sparse': module.k_sparse_write, 'surprise_scale': module.surprise_scale_factor,
                  'action_std': module.action_std
              })
         task_head_params = {
             'gru_params': [p.detach().clone() for p in self.task_head.gru.parameters()],
             'cls_pred_params': [p.detach().clone() for p in self.task_head.cls_pred_layer.parameters()],
             'qm': self.task_head.qm.detach().clone(), 'sm_baseline': self.task_head.sm_log_baseline.detach().clone(),
             'k_sparse': self.task_head.k_sparse_write, 'surprise_scale': self.task_head.surprise_scale_factor,
             'hidden_dim': self.task_head.module_hidden_dim, 'cls_dim': self.cls_dim,
             'action_std': self.task_head.action_std
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
                      log_surprise_diff = 0.0
                      influence_scalar_am = 1.0 + 1.0 * torch.tanh(torch.tensor(params['surprise_scale'] * log_surprise_diff, device=self.device))
                      influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                      intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                      action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)
                      write_vector_im_sparse_vals = sparsify_vector(action_mean, params['k_sparse'])
                      sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0); sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                      if sparse_indices.numel() > 0: sum_im += torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=action_mean.dtype)

                 params = task_head_params
                 try:
                     gru_layer = nn.GRU(params['cls_dim'], params['hidden_dim'], batch_first=True).to(self.device)
                     gru_layer.load_state_dict({k: v for k, v in zip(gru_layer.state_dict().keys(), params['gru_params'])})
                     cls_pred_layer = nn.Linear(params['hidden_dim'], params['cls_dim']).to(self.device)
                     cls_pred_layer.load_state_dict({'weight': params['cls_pred_params'][0], 'bias': params['cls_pred_params'][1]})
                     gru_output, _ = gru_layer(state_t_sparse.to_dense().unsqueeze(0).unsqueeze(0))
                     last_hidden = gru_output[:, -1, :]
                     pred_cls = cls_pred_layer(last_hidden).squeeze(0)
                     pred_cls = torch.clamp(pred_cls, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
                 except Exception as e:
                     pred_cls = torch.zeros(self.cls_dim, device=self.device)
                 vm = pred_cls; gate_raw_score = params['qm'] * state_t_sparse.to_dense(); gate_activation_gm = torch.sigmoid(gate_raw_score / 1.0)
                 log_surprise_diff = 0.0
                 influence_scalar_am = 1.0 + 1.0 * torch.tanh(torch.tensor(params['surprise_scale'] * log_surprise_diff, device=self.device))
                 influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                 intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                 action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)
                 write_vector_im_sparse_vals = sparsify_vector(action_mean, params['k_sparse'])
                 sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0); sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                 if sparse_indices.numel() > 0: sum_im += torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=action_mean.dtype)

                 gamma_t = gamma_val; mmodt = self._init_cls()
                 next_state_sparse = self.cls_update_rule(state_t_sparse, sum_im.coalesce(), mmodt, gamma_t, fixed_noise_std)
                 next_state_dense = next_state_sparse.to_dense()
                 if not torch.all(torch.isfinite(next_state_dense)):
                     next_state_dense = torch.nan_to_num(next_state_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)
             return next_state_dense
         return dynamics_map

    def step(self, current_step_num, task_input, estimate_le=False, true_answer_c=None):
        """Performs one full step using LOG-SURPRISE and COMBINED module loss."""

        a, op_idx, b, _ = task_input

        if not torch.all(torch.isfinite(self.ct.values())):
             print(f"Warning: CLS became non-finite at step {current_step_num}. Resetting CLS and states.")
             self.ct = self._init_cls()
             self.gt_log_ema = None; self.lambda_max_ema = None
             for module in self.pred_modules: module.sm_log_baseline.fill_(0.0); module.last_log_surprise_sm_cls = torch.tensor(0.0, device=self.device)
             self.task_head.sm_log_baseline.fill_(0.0); self.task_head.last_log_surprise_sm_cls = torch.tensor(0.0, device=self.device)
             self.task_head.last_gru_hidden_state = None
             self.meta_model.last_gru_hidden_state = None

        ct_prev_dense = self.ct.to_dense().detach()

        # --- 1. Module Predictions (Needs grad for fm update) ---
        all_modules = self.pred_modules + [self.task_head]
        # module_predictions_sm = {} # Not needed explicitly
        for i, module in enumerate(all_modules):
            _ = module.predict(self.ct) # Stores prediction internally

        # --- 2. Meta-Model Regulation ---
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.gt_log_ema, self.lambda_max_ema)

        # --- 3. Module Write Vectors (Needs grad for qm update) ---
        sum_im = self._init_cls()
        i_math = self.math_encoder(a, op_idx, b)
        sum_im += i_math
        for i, module in enumerate(all_modules):
             im = module.generate_write_vector(self.ct) # Stores log_prob internally
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual LOG-Surprises (Needs grad for fm update) ---
        # current_log_surprises_sm_grad = {} # Not needed explicitly
        current_log_surprises_sm_detached = {} # Store detached surprise for reward/logging
        log_surprises_sm_cls_detached = [] # Store detached CLS surprise for logging
        raw_surprises_sm_cls_detached = [] # Store detached raw CLS surprise for logging
        global_log_surprise_gt_sum = 0.0
        valid_surprises = 0
        for i, module in enumerate(all_modules):
             _ = module.calculate_surprise(ct_plus_1, true_task_output=None) # Stores grad surprise internally, updates detached state
             sm_log_detached = module.last_log_surprise_sm_cls # Use the detached CLS part
             current_log_surprises_sm_detached[i] = sm_log_detached
             log_surprises_sm_cls_detached.append(sm_log_detached)
             raw_surprises_sm_cls_detached.append(module.last_raw_surprise_sm_cls)
             if math.isfinite(sm_log_detached.item()):
                 global_log_surprise_gt_sum += sm_log_detached.item(); valid_surprises += 1

        global_log_surprise_gt = (global_log_surprise_gt_sum / valid_surprises) if valid_surprises > 0 else float('nan')

        # --- 6. Estimate LE (Optional) ---
        lambda_max_estimate = None
        if estimate_le and torch.all(torch.isfinite(ct_prev_dense)):
            dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
            lambda_max_estimate = estimate_lyapunov_exponent(dynamics_map_for_le, ct_prev_dense, device=self.device)
            if lambda_max_estimate is not None and not math.isfinite(lambda_max_estimate):
                 lambda_max_estimate = None

        # --- 7. Update Meta-Model EMA Inputs ---
        current_gt_log_val = global_log_surprise_gt
        if math.isfinite(current_gt_log_val):
            if self.gt_log_ema is None: self.gt_log_ema = current_gt_log_val
            else: self.gt_log_ema = (1 - self.ema_alpha) * current_gt_log_val + self.ema_alpha * self.gt_log_ema

        current_lambda_val = lambda_max_estimate
        if current_lambda_val is not None and math.isfinite(current_lambda_val):
             effective_lambda = current_lambda_val
        else:
             effective_lambda = self.meta_model.stability_target # Use target if LE invalid

        if self.lambda_max_ema is None: self.lambda_max_ema = effective_lambda
        else: self.lambda_max_ema = (1 - self.ema_alpha) * effective_lambda + self.ema_alpha * self.lambda_max_ema


        # --- 8. Trigger Meta-Model Learning ---
        meta_loss_item, meta_grad_norm = self.meta_model.learn(self.gt_log_ema, self.lambda_max_ema)

        # --- 9. Trigger Module Learning (Combined Loss) ---
        module_total_losses = []
        module_policy_losses = []
        module_pred_losses = []
        module_entropies = []
        module_grad_norms = []
        for i, module in enumerate(all_modules):
            local_sm_log_detached = current_log_surprises_sm_detached.get(i, None)
            total_loss, policy_loss, pred_loss, entropy, grad_norm = module.learn(local_sm_log_detached)
            module_total_losses.append(total_loss)
            module_policy_losses.append(policy_loss)
            module_pred_losses.append(pred_loss)
            module_entropies.append(entropy)
            module_grad_norms.append(grad_norm)

        # --- 10. Update State ---
        self.ct = ct_plus_1

        # --- 11. Return Metrics (Combined Loss, Task Disabled) ---
        finite_sm_log_cls = [s.item() for s in log_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_sm_raw_cls = [s.item() for s in raw_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_total_losses = [l for l in module_total_losses if l is not None and math.isfinite(l)]
        finite_policy_losses = [l for l in module_policy_losses if l is not None and math.isfinite(l)]
        finite_pred_losses = [l for l in module_pred_losses if l is not None and math.isfinite(l)]
        finite_entropies = [e for e in module_entropies if e is not None and math.isfinite(e)]
        finite_module_grads = [g for g in module_grad_norms if g is not None and math.isfinite(g)]

        metrics = {
            "Gt_log": global_log_surprise_gt if math.isfinite(global_log_surprise_gt) else float('nan'),
            "Gt_log_EMA": self.gt_log_ema,
            "Sm_log_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_log_std": np.std(finite_sm_log_cls) if len(finite_sm_log_cls) > 1 else 0.0,
            "Sm_log_cls_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_raw_cls_avg": np.mean(finite_sm_raw_cls) if finite_sm_raw_cls else float('nan'),
            "lambda_max_est": lambda_max_estimate,
            "lambda_max_EMA": self.lambda_max_ema,
            "gamma_t": gamma_t,
            "noise_std": noise_std_dev,
            "module_loss_avg": np.mean(finite_total_losses) if finite_total_losses else float('nan'),
            "module_policy_loss_avg": np.mean(finite_policy_losses) if finite_policy_losses else float('nan'),
            "module_pred_loss_avg": np.mean(finite_pred_losses) if finite_pred_losses else float('nan'),
            "meta_loss": meta_loss_item if math.isfinite(meta_loss_item) else float('nan'),
            "module_entropy_avg": np.mean(finite_entropies) if finite_entropies else float('nan'),
            "module_grad_norm_avg": np.mean(finite_module_grads) if finite_module_grads else float('nan'),
            "meta_grad_norm": meta_grad_norm if math.isfinite(meta_grad_norm) else float('nan'),
            "cls_norm": torch.linalg.norm(self.ct.to_dense()).item() if torch.all(torch.isfinite(self.ct.values())) else float('nan'),
            "cls_density": self.ct._nnz() / self.cls_dim if self.ct._nnz() is not None and self.cls_dim > 0 else float('nan')
        }
        return metrics