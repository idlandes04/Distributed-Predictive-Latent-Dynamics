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
DEFAULT_ACTION_STD = 0.1 # MODIFIED Rev 9: Reduced default action std
CLS_NORM_CLIP_VAL = 100.0
MODULE_OUTPUT_CLIP_VAL = 100.0
ACTION_MEAN_CLIP_VAL = 5.0
GRAD_CLIP_NORM = 1.0
META_REWARD_CLIP = 10.0
DEFAULT_PREDICTION_LOSS_WEIGHT = 1.0
DEFAULT_TASK_LOSS_WEIGHT = 1.0
TASK_REWARD_SCALE_FACTOR = 0.01 # MODIFIED Rev 9: Added scaling for task reward


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


# --- Predictive Module (Base Class - Added step to learn) ---
class PredictiveModule(nn.Module):
    """
    Base class for DPLD modules: Read, Predict, Write.
    Uses combined loss: Policy Loss (for qm) + Prediction Loss (for fm).
    Uses simple reward = -local_log_surprise_sm_cls for policy loss.
    Uses LOG-SURPRISE internally and for influence scaling.
    Includes stability clamping, action mean clipping, and weight decay.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0,
                 surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT,
                 task_loss_weight=0.0, # Ignored by base class learn/surprise
                 weight_decay=1e-5,
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
        # self.task_loss_weight is unused here

        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)

        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01)

        self.register_buffer('sm_log_baseline', torch.tensor(0.0, device=device))
        self.last_prediction_ct_plus_1 = None
        self.last_log_surprise_sm = None # Grad-enabled combined loss value
        self.last_log_surprise_sm_cls = torch.tensor(0.0, device=device) # Detached CLS surprise
        self.last_raw_surprise_sm_cls = None
        self.last_log_prob = None
        self.last_action_dist = None

        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate, weight_decay=weight_decay)

    def predict(self, ct):
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        prediction = self.fm(ct_dense.detach())
        prediction = torch.clamp(prediction, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_prediction_ct_plus_1 = prediction
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates local LOG-SURPRISE Sm (only CLS part for base module)."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             self.last_log_surprise_sm = torch.tensor(0.0, device=self.device)
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             return self.last_log_surprise_sm

        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            sm_cls_raw = torch.tensor(1e6, device=self.device)
            if torch.is_grad_enabled(): sm_cls_raw = sm_cls_raw.requires_grad_() # Add grad if needed for loss
        else:
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach()

        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)
        self.last_log_surprise_sm_cls = sm_cls_log.detach()

        # Combined surprise for loss = weighted CLS surprise
        combined_log_surprise = self.prediction_loss_weight * sm_cls_log

        if not math.isfinite(combined_log_surprise.item()):
             self.last_log_surprise_sm = torch.tensor(self.sm_log_baseline.item() + 1.0, device=self.device)
        else:
            self.last_log_surprise_sm = combined_log_surprise
            with torch.no_grad():
                update_val = self.last_log_surprise_sm_cls if torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * update_val

        return self.last_log_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse write vector Im using LOG-SURPRISE (CLS part) for alpha_m."""
        if self.last_prediction_ct_plus_1 is None:
             vm = torch.zeros(self.cls_dim, device=self.device)
             current_log_surprise_detached = self.sm_log_baseline.detach()
        else:
             current_log_surprise_detached = self.last_log_surprise_sm_cls if self.last_log_surprise_sm_cls is not None and torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline.detach()
             vm = self.last_prediction_ct_plus_1.detach()

        log_surprise_diff = current_log_surprise_detached - self.sm_log_baseline.detach()
        ct_dense = ct.to_dense() if ct.is_sparse else ct

        tau_g = 1.0
        gate_raw_score = self.qm * ct_dense.detach()
        gate_activation_gm = torch.sigmoid(gate_raw_score / tau_g)

        alpha_base = 1.0; alpha_scale = 1.0
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * log_surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0).detach()

        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
        action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

        if not torch.all(torch.isfinite(action_mean)):
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=ACTION_MEAN_CLIP_VAL, neginf=-ACTION_MEAN_CLIP_VAL)

        action_std_tensor = torch.full_like(action_mean, fill_value=self.action_std)

        try:
            dist = Normal(action_mean, action_std_tensor)
            self.last_action_dist = dist
            dense_write_vector_sampled = dist.sample()
            self.last_log_prob = dist.log_prob(dense_write_vector_sampled.detach()).sum()
        except ValueError as e:
             print(f"ERROR creating Normal distribution in {self.__class__.__name__}: {e}")
             dense_write_vector_sampled = torch.zeros_like(action_mean)
             self.last_log_prob = None
             self.last_action_dist = None

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

    def learn(self, local_log_surprise_sm_cls_detached, step): # MODIFIED Rev 9: Added step
        """
        Updates base module parameters (fm and qm) using combined loss.
        Policy reward uses negative CLS surprise.
        Prediction loss uses weighted CLS surprise.
        """
        policy_loss_item = 0.0
        prediction_loss_item = 0.0
        entropy_item = 0.0
        grad_norm = 0.0

        # --- 1. Calculate Policy Loss Component (for qm) ---
        policy_loss = torch.tensor(0.0, device=self.device)
        if self.last_log_prob is not None and self.last_action_dist is not None and \
           local_log_surprise_sm_cls_detached is not None and math.isfinite(local_log_surprise_sm_cls_detached.item()):

            reward_tensor = -local_log_surprise_sm_cls_detached # Use CLS surprise for reward

            entropy_tensor = torch.tensor(0.0, device=self.device)
            try:
                if self.last_action_dist is not None:
                    entropy_tensor = self.last_action_dist.entropy().sum()
                    if not torch.isfinite(entropy_tensor): entropy_tensor = torch.tensor(0.0, device=self.device)
                entropy_item = entropy_tensor.item()

                # MODIFIED Rev 9: Added entropy debug print
                if step % 500 == 1: # Print occasionally
                    mean_check = self.last_action_dist.mean.mean().item()
                    std_check = self.last_action_dist.scale.mean().item()
                    print(f"Step {step} - {self.__class__.__name__}: Entropy={entropy_item:.2f}, Mean={mean_check:.4f}, Std={std_check:.4f}")

            except Exception as e:
                 print(f"Warning: Entropy calculation/debug error in PredictiveModule: {e}")
                 entropy_tensor = torch.tensor(0.0, device=self.device); entropy_item = 0.0

            entropy_term = self.entropy_coeff * entropy_tensor

            if torch.isfinite(self.last_log_prob):
                policy_loss = -(reward_tensor * self.last_log_prob + entropy_term)
                if not torch.isfinite(policy_loss):
                    policy_loss = torch.tensor(0.0, device=self.device)
            else:
                policy_loss = torch.tensor(0.0, device=self.device)
        policy_loss_item = policy_loss.item()

        # --- 2. Calculate Prediction Loss Component (for fm) ---
        prediction_loss = torch.tensor(0.0, device=self.device)
        # self.last_log_surprise_sm has grad history back to fm (set in calculate_surprise)
        if self.last_log_surprise_sm is not None and torch.is_tensor(self.last_log_surprise_sm) and self.last_log_surprise_sm.requires_grad:
            if torch.isfinite(self.last_log_surprise_sm):
                prediction_loss = self.last_log_surprise_sm # Already weighted in calculate_surprise
            else:
                prediction_loss = torch.tensor(0.0, device=self.device)
        prediction_loss_item = prediction_loss.item()

        # --- 3. Combine Losses ---
        total_loss = policy_loss + prediction_loss

        # --- 4. Backpropagation and Optimizer Step ---
        if torch.isfinite(total_loss) and total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()

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
            grad_norm = 0.0
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN/Inf total_loss in {self.__class__.__name__}.learn(). Policy: {policy_loss_item}, Pred: {prediction_loss_item}")

        self.last_log_prob = None
        self.last_action_dist = None

        return total_loss.item(), policy_loss_item, prediction_loss_item, entropy_item, grad_norm


# --- Task Head Module (Overrides calculate_surprise and learn, uses specific LR) ---
class TaskHead(PredictiveModule):
    """
    Specialized PredictiveModule for the arithmetic task.
    Prediction Loss = weight_cls * Sm_cls + weight_task * Sm_task
    Policy Reward = SCALED -Sm_task (detached)
    Uses LOG-SURPRISE. Internal model `fm` uses a GRU.
    Includes weight decay. Uses specific learning rate.
    """
    # MODIFIED Rev 9: Added taskhead_lr
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, taskhead_lr, module_lr, # Added taskhead_lr, module_lr is for base init compatibility
                 entropy_coeff=0.01,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT, # Weight for CLS part
                 task_loss_weight=DEFAULT_TASK_LOSS_WEIGHT, # Weight for Task part
                 weight_decay=1e-5,
                 gru_layers=1, device='cpu'):
        # Initialize PredictiveModule - pass weights, use module_lr for base init
        super().__init__(cls_dim, module_hidden_dim, k_sparse_write, module_lr, # Use module_lr here
                         entropy_coeff, surprise_scale_factor, surprise_baseline_ema, action_std,
                         prediction_loss_weight=prediction_loss_weight,
                         task_loss_weight=task_loss_weight, # Pass task weight
                         weight_decay=weight_decay, device=device)

        # Store task loss weight specifically for TaskHead
        self.task_loss_weight = task_loss_weight

        # --- fm: Prediction Network (GRU + Linear layers) ---
        self.gru = nn.GRU(cls_dim, module_hidden_dim, num_layers=gru_layers, batch_first=True).to(device)
        self.task_out_layer = nn.Linear(module_hidden_dim, 1).to(device)
        self.cls_pred_layer = nn.Linear(module_hidden_dim, cls_dim).to(device)

        # MODIFIED Rev 9: Re-define optimizer using taskhead_lr
        self.optimizer = optim.Adam(
            list(self.gru.parameters()) +
            list(self.task_out_layer.parameters()) +
            list(self.cls_pred_layer.parameters()) +
            [self.qm], # qm is inherited
            lr=taskhead_lr, # Use specific taskhead_lr
            weight_decay=weight_decay
        )

        self.last_gru_hidden_state = None
        self.last_task_prediction = None # Stores detached task prediction
        # Inherits last_log_surprise_sm_cls, last_raw_surprise_sm_cls from base
        # Add specific storage for task surprise
        self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device) # Detached task surprise
        self.last_raw_surprise_sm_task = None

    def predict(self, ct):
        """Predicts both next CLS state and task answer."""
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        ct_input = ct_dense.detach().unsqueeze(0).unsqueeze(0)
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None

        if hidden_state_input is not None and hidden_state_input.shape[1] != ct_input.shape[0]:
             hidden_state_input = None

        gru_output, next_hidden_state = self.gru(ct_input, hidden_state_input)
        self.last_gru_hidden_state = next_hidden_state.detach()

        last_hidden = gru_output[:, -1, :]

        predicted_answer = self.task_out_layer(last_hidden).squeeze()
        self.last_task_prediction = torch.clamp(predicted_answer, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL).detach()

        predicted_cls = self.cls_pred_layer(last_hidden).squeeze(0)
        self.last_prediction_ct_plus_1 = torch.clamp(predicted_cls, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)

        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates combined LOG-SURPRISE Sm for TaskHead."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None or self.last_task_prediction is None:
             self.last_log_surprise_sm = torch.tensor(0.0, device=self.device)
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device)
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             self.last_raw_surprise_sm_task = torch.tensor(float('nan'), device=self.device)
             return self.last_log_surprise_sm

        # --- Calculate CLS Surprise ---
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            sm_cls_raw = torch.tensor(1e6, device=self.device)
            if torch.is_grad_enabled(): sm_cls_raw = sm_cls_raw.requires_grad_()
        else:
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach()
        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)
        self.last_log_surprise_sm_cls = sm_cls_log.detach()

        # --- Calculate Task Surprise ---
        if true_task_output is not None and torch.all(torch.isfinite(self.last_task_prediction)) and torch.isfinite(true_task_output):
            sm_task_raw = F.mse_loss(self.last_task_prediction.detach(), true_task_output.float().detach(), reduction='mean') # Use detached pred for raw calc
            # Need grad-enabled version for loss calculation
            # Rerun task prediction part with grad
            # Ensure ct_input is correctly shaped for GRU (batch, seq, features)
            ct_dense_for_rerun = self.last_prediction_ct_plus_1.unsqueeze(0).unsqueeze(0)
            hidden_state_input_rerun = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
            if hidden_state_input_rerun is not None and hidden_state_input_rerun.shape[1] != ct_dense_for_rerun.shape[0]:
                hidden_state_input_rerun = None
            gru_output_rerun, _ = self.gru(ct_dense_for_rerun, hidden_state_input_rerun)
            last_hidden_rerun = gru_output_rerun[:, -1, :]
            predicted_answer_grad = self.task_out_layer(last_hidden_rerun).squeeze()

            sm_task_raw_grad = F.mse_loss(predicted_answer_grad, true_task_output.float().detach(), reduction='mean')
        else:
            sm_task_raw = torch.tensor(1e6, device=self.device)
            sm_task_raw_grad = torch.tensor(1e6, device=self.device)
            if torch.is_grad_enabled(): sm_task_raw_grad = sm_task_raw_grad.requires_grad_()

        self.last_raw_surprise_sm_task = sm_task_raw.detach()
        sm_task_log = torch.log1p(sm_task_raw_grad + EPSILON) # Use grad version for log surprise
        self.last_log_surprise_sm_task = sm_task_log.detach() # Store detached log task surprise

        # --- Combine for Prediction Loss ---
        # Use grad-enabled log surprises
        combined_log_surprise = self.prediction_loss_weight * sm_cls_log + \
                                self.task_loss_weight * sm_task_log

        if not math.isfinite(combined_log_surprise.item()):
             self.last_log_surprise_sm = torch.tensor(self.sm_log_baseline.item() + 1.0, device=self.device) # No grad history
        else:
            self.last_log_surprise_sm = combined_log_surprise
            # Update baseline using only CLS surprise
            with torch.no_grad():
                update_val = self.last_log_surprise_sm_cls if torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * update_val

        return self.last_log_surprise_sm

    def get_task_prediction(self):
        return self.last_task_prediction

    def learn(self, _, step): # MODIFIED Rev 9: Added step, first arg ignored
        """
        Updates TaskHead parameters (fm=GRU/Linear and qm) using combined loss.
        Uses SCALED TASK surprise for policy reward.
        """
        policy_loss_item = 0.0
        prediction_loss_item = 0.0
        entropy_item = 0.0
        grad_norm = 0.0

        # --- 1. Calculate Policy Loss Component (for qm) ---
        policy_loss = torch.tensor(0.0, device=self.device)
        # Use DETACHED negative TASK log surprise as reward
        if self.last_log_prob is not None and self.last_action_dist is not None and \
           self.last_log_surprise_sm_task is not None and math.isfinite(self.last_log_surprise_sm_task.item()):

            # MODIFIED Rev 9: Scale the reward
            raw_reward_tensor = -self.last_log_surprise_sm_task.detach() # Use task surprise
            reward_tensor = raw_reward_tensor * TASK_REWARD_SCALE_FACTOR # Scale down

            entropy_tensor = torch.tensor(0.0, device=self.device)
            try:
                if self.last_action_dist is not None: # Check if dist is valid
                    entropy_tensor = self.last_action_dist.entropy().sum()
                    if not torch.isfinite(entropy_tensor): entropy_tensor = torch.tensor(0.0, device=self.device)
                entropy_item = entropy_tensor.item()

                # MODIFIED Rev 9: Added entropy debug print
                if step % 500 == 1: # Print occasionally
                    mean_check = self.last_action_dist.mean.mean().item()
                    std_check = self.last_action_dist.scale.mean().item()
                    print(f"Step {step} - {self.__class__.__name__}: Entropy={entropy_item:.2f}, Mean={mean_check:.4f}, Std={std_check:.4f}")

            except Exception as e:
                 print(f"Warning: Entropy calculation/debug error in TaskHead: {e}")
                 entropy_tensor = torch.tensor(0.0, device=self.device); entropy_item = 0.0

            entropy_term = self.entropy_coeff * entropy_tensor

            if torch.isfinite(self.last_log_prob):
                policy_loss = -(reward_tensor * self.last_log_prob + entropy_term)
                if not torch.isfinite(policy_loss):
                    policy_loss = torch.tensor(0.0, device=self.device)
            else:
                policy_loss = torch.tensor(0.0, device=self.device)
        policy_loss_item = policy_loss.item()

        # --- 2. Calculate Prediction Loss Component (for fm=GRU/Linear) ---
        prediction_loss = torch.tensor(0.0, device=self.device)
        # self.last_log_surprise_sm has grad history back to fm (set in calculate_surprise)
        # and now includes the weighted task surprise component.
        if self.last_log_surprise_sm is not None and torch.is_tensor(self.last_log_surprise_sm) and self.last_log_surprise_sm.requires_grad:
            if torch.isfinite(self.last_log_surprise_sm):
                prediction_loss = self.last_log_surprise_sm # Already weighted in calculate_surprise
            else:
                prediction_loss = torch.tensor(0.0, device=self.device)
        prediction_loss_item = prediction_loss.item()

        # --- 3. Combine Losses ---
        total_loss = policy_loss + prediction_loss

        # --- 4. Backpropagation and Optimizer Step ---
        if torch.isfinite(total_loss) and total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()

            params_to_clip = list(self.gru.parameters()) + \
                             list(self.task_out_layer.parameters()) + \
                             list(self.cls_pred_layer.parameters()) + \
                             [self.qm]

            total_norm_sq = 0.0
            for p in params_to_clip:
                if p.grad is not None:
                     param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                     if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
            grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

            nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
            self.optimizer.step()
        else:
            grad_norm = 0.0
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN/Inf total_loss in TaskHead.learn(). Policy: {policy_loss_item}, Pred: {prediction_loss_item}")

        self.last_log_prob = None
        self.last_action_dist = None

        return total_loss.item(), policy_loss_item, prediction_loss_item, entropy_item, grad_norm


# --- Meta-Model (Unchanged from Rev 7) ---
class MetaModel(nn.Module):
    """
    Implements DPLD Meta-Model for stability regulation.
    LEARNING IS ENABLED. Uses smoothed inputs (EMA Gt, EMA lambda_max) and clips reward.
    Includes weight decay. Stability target can be set high to ignore LE.
    """
    def __init__(self, meta_input_dim, meta_hidden_dim, learning_rate, cls_dim,
                 gamma_min=0.01, gamma_max=0.2, stability_target=-0.01, # Default target closer to 0
                 weight_decay=1e-5,
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
            self.last_gru_hidden_state = next_hidden_state.detach()
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

        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        gru_output_rerun, _ = self.gru(self.last_meta_input_smoothed, hidden_state_input)
        last_hidden_rerun = gru_output_rerun[:, -1, :]
        gamma_offset_signal_mean = self.output_layer(last_hidden_rerun).squeeze()

        meta_action_std = torch.tensor(0.1, device=self.device)
        try:
            meta_action_dist = Normal(gamma_offset_signal_mean, meta_action_std)
            log_prob_meta_action = meta_action_dist.log_prob(self.last_gamma_offset_signal)
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


# --- DPLD System (Added taskhead_lr, simplified dynamics map, pass step to learn) ---
class DPLDSystem(nn.Module):
    """
    Main DPLD system coordinating CLS, Modules, Meta-Model, and Task components.
    Enables task-specific loss and reward for TaskHead.
    Uses LOG-SURPRISE for internal calculations.
    Includes EMA smoothing, weight decay, action mean clipping.
    MODIFIED Rev 9: Simplified dynamics map, passes step to learn.
    """
    # MODIFIED Rev 9: Added taskhead_lr
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, taskhead_lr, noise_std_dev_schedule, # Added taskhead_lr
                 env, embedding_dim,
                 entropy_coeff=0.01,
                 ema_alpha=0.99,
                 gamma_min=0.01, gamma_max=0.2, stability_target=-0.01,
                 action_std=DEFAULT_ACTION_STD,
                 prediction_loss_weight=DEFAULT_PREDICTION_LOSS_WEIGHT,
                 task_loss_weight=DEFAULT_TASK_LOSS_WEIGHT,
                 weight_decay=1e-5,
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

        # MODIFIED Rev 9: Pass taskhead_lr
        self.task_head = TaskHead(cls_dim, module_hidden_dim, k_sparse_write, taskhead_lr, module_lr, # Pass both LRs
                                  entropy_coeff=entropy_coeff,
                                  action_std=action_std,
                                  prediction_loss_weight=prediction_loss_weight,
                                  task_loss_weight=task_loss_weight,
                                  weight_decay=weight_decay, device=device).to(device)

        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                             entropy_coeff=entropy_coeff,
                             action_std=action_std,
                             prediction_loss_weight=prediction_loss_weight,
                             task_loss_weight=0.0, # Base modules ignore this
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
         """
         Returns a function for LE estimation.
         MODIFIED Rev 9: Simplified to use current module parameters directly.
         """
         gamma_val = float(fixed_gamma)
         all_modules = self.pred_modules + [self.task_head]

         def dynamics_map(state_t_dense):
             # Ensure input requires grad for JVP
             state_t_dense = state_t_dense.clone().requires_grad_(True)

             if not torch.all(torch.isfinite(state_t_dense)):
                 print("Warning (LE Dynamics Map): Input state non-finite.")
                 return state_t_dense.detach() # Return detached if input is bad

             state_t_sparse = state_t_dense.detach().to_sparse_coo() # Detach before converting to sparse
             sum_im = self._init_cls()

             # --- Simulate Module Writes using current parameters ---
             # Need torch.no_grad() here because we don't want the *module* parameters
             # updated during this simulation, but the final result MUST depend on state_t_dense.
             with torch.no_grad():
                 for module in all_modules:
                     # Simulate prediction
                     # We need the prediction vm, but detached from module parameters
                     # We can call predict and detach the result
                     pred_cls_detached = module.predict(state_t_sparse).detach()
                     vm = pred_cls_detached

                     # Simulate gating and influence scaling (uses detached state/surprise)
                     current_log_surprise_detached = module.sm_log_baseline.detach() # Use baseline for LE
                     log_surprise_diff = current_log_surprise_detached - module.sm_log_baseline.detach()
                     gate_raw_score = module.qm.detach() * state_t_dense # Use qm but detached
                     gate_activation_gm = torch.sigmoid(gate_raw_score / 1.0) # tau_g = 1.0
                     influence_scalar_am = 1.0 + 1.0 * torch.tanh(module.surprise_scale_factor * log_surprise_diff)
                     influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)

                     # Calculate action mean (detached from module params)
                     intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                     action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

                     # Use the mean action for dynamics map (no sampling)
                     write_vector_im_sparse_vals = sparsify_vector(action_mean, module.k_sparse_write)
                     sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
                     sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]
                     if sparse_indices.numel() > 0:
                         im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=action_mean.dtype)
                         sum_im += im_sparse

             # --- CLS Update ---
             # The update rule itself should allow gradients from state_t_dense
             # (via the (1-gamma)*ct term and potentially the gating term if qm wasn't detached)
             gamma_t = gamma_val
             mmodt = self._init_cls() # Assume no meta-model influence for LE calculation
             noise_et = torch.randn(self.cls_dim, device=self.device) * fixed_noise_std

             # Re-calculate the parts that depend on state_t_dense *with* grad enabled
             decayed_ct_grad = (1.0 - gamma_t) * state_t_dense # Depends on input
             sum_im_dense_grad = sum_im.to_dense() # This part is detached from module params

             # --- Potential issue: Does gating depend on state_t_dense with grad? ---
             # Let's recalculate gating explicitly to ensure grad flow from state_t_dense
             gating_dependent_sum_im = torch.zeros_like(state_t_dense)
             with torch.no_grad(): # Get detached predictions again
                 module_preds = [m.predict(state_t_sparse).detach() for m in all_modules]
             for i, module in enumerate(all_modules):
                 vm = module_preds[i]
                 gate_raw_score_grad = module.qm.detach() * state_t_dense # Use detached qm, grad flows from state_t_dense
                 gate_activation_gm_grad = torch.sigmoid(gate_raw_score_grad / 1.0)
                 # Use detached surprise for influence
                 current_log_surprise_detached = module.sm_log_baseline.detach()
                 log_surprise_diff = current_log_surprise_detached - module.sm_log_baseline.detach()
                 influence_scalar_am = 1.0 + 1.0 * torch.tanh(module.surprise_scale_factor * log_surprise_diff)
                 influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                 # Calculate action mean using grad-enabled gating
                 intermediate_write_vector_grad = influence_scalar_am * (gate_activation_gm_grad * vm)
                 action_mean_grad = torch.clamp(intermediate_write_vector_grad, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)
                 # Sparsify (non-differentiable, but okay as we use the mean)
                 write_vals_grad = sparsify_vector(action_mean_grad, module.k_sparse_write)
                 gating_dependent_sum_im += write_vals_grad # Add sparse vector contribution

             # Combine terms
             next_state_dense = decayed_ct_grad + gating_dependent_sum_im + noise_et

             if not torch.all(torch.isfinite(next_state_dense)):
                 print("Warning (LE Dynamics Map): Output state non-finite.")
                 next_state_dense = torch.nan_to_num(next_state_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)

             # Apply norm clipping if needed
             if self.clip_cls_norm:
                 norm = torch.linalg.norm(next_state_dense)
                 if norm > CLS_NORM_CLIP_VAL:
                     next_state_dense = next_state_dense * (CLS_NORM_CLIP_VAL / (norm + EPSILON))

             # Return the result, ensuring it still requires grad w.r.t. input
             return next_state_dense

         return dynamics_map


    def step(self, current_step_num, task_input, estimate_le=False, true_answer_c=None):
        """Performs one full step using LOG-SURPRISE and enabling TaskHead learning."""

        a, op_idx, b, _ = task_input # True answer c is passed separately

        if not torch.all(torch.isfinite(self.ct.values())):
             print(f"Warning: CLS became non-finite at step {current_step_num}. Resetting CLS and states.")
             self.ct = self._init_cls()
             self.gt_log_ema = None; self.lambda_max_ema = None
             for module in self.pred_modules: module.sm_log_baseline.fill_(0.0); module.last_log_surprise_sm_cls = torch.tensor(0.0, device=self.device)
             self.task_head.sm_log_baseline.fill_(0.0); self.task_head.last_log_surprise_sm_cls = torch.tensor(0.0, device=self.device); self.task_head.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device)
             self.task_head.last_gru_hidden_state = None
             self.meta_model.last_gru_hidden_state = None

        ct_prev_dense = self.ct.to_dense().detach() # Detach here for LE

        # --- 1. Module Predictions ---
        all_modules = self.pred_modules + [self.task_head]
        for i, module in enumerate(all_modules):
            _ = module.predict(self.ct)

        # --- 2. Meta-Model Regulation ---
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.gt_log_ema, self.lambda_max_ema)

        # --- 3. Module Write Vectors ---
        sum_im = self._init_cls()
        i_math = self.math_encoder(a, op_idx, b)
        sum_im += i_math
        for i, module in enumerate(all_modules):
             im = module.generate_write_vector(self.ct)
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual LOG-Surprises ---
        log_surprises_sm_cls_detached = []
        raw_surprises_sm_cls_detached = []
        global_log_surprise_gt_sum = 0.0
        valid_surprises = 0
        for i, module in enumerate(all_modules):
             # Pass true_answer_c only to TaskHead's surprise calculation
             true_task = true_answer_c if module is self.task_head else None
             _ = module.calculate_surprise(ct_plus_1, true_task_output=true_task)
             sm_log_cls_detached = module.last_log_surprise_sm_cls # Use the detached CLS part
             log_surprises_sm_cls_detached.append(sm_log_cls_detached)
             raw_surprises_sm_cls_detached.append(module.last_raw_surprise_sm_cls)
             if math.isfinite(sm_log_cls_detached.item()):
                 global_log_surprise_gt_sum += sm_log_cls_detached.item(); valid_surprises += 1

        global_log_surprise_gt = (global_log_surprise_gt_sum / valid_surprises) if valid_surprises > 0 else float('nan')

        # --- 6. Estimate LE (Optional) ---
        lambda_max_estimate = None
        if estimate_le and torch.all(torch.isfinite(ct_prev_dense)):
            try:
                dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
                # Ensure input to LE has requires_grad=False, LE function handles requires_grad=True internally
                lambda_max_estimate = estimate_lyapunov_exponent(dynamics_map_for_le, ct_prev_dense.detach(), device=self.device)
                if lambda_max_estimate is not None and not math.isfinite(lambda_max_estimate):
                     lambda_max_estimate = None
            except Exception as le_error:
                 print(f"Error during LE estimation call: {le_error}")
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
             effective_lambda = self.meta_model.stability_target

        if self.lambda_max_ema is None: self.lambda_max_ema = effective_lambda
        else: self.lambda_max_ema = (1 - self.ema_alpha) * effective_lambda + self.ema_alpha * self.lambda_max_ema

        # --- 8. Trigger Meta-Model Learning ---
        meta_loss_item, meta_grad_norm = self.meta_model.learn(self.gt_log_ema, self.lambda_max_ema)

        # --- 9. Trigger Module Learning ---
        module_total_losses = []
        module_policy_losses = []
        module_pred_losses = []
        module_entropies = []
        module_grad_norms = []
        for i, module in enumerate(all_modules):
            # Base modules use detached CLS surprise for reward
            # TaskHead uses scaled detached TASK surprise for reward (handled internally)
            local_sm_log_cls_detached = log_surprises_sm_cls_detached[i]
            # MODIFIED Rev 9: Pass step to learn
            total_loss, policy_loss, pred_loss, entropy, grad_norm = module.learn(local_sm_log_cls_detached, current_step_num)
            module_total_losses.append(total_loss)
            module_policy_losses.append(policy_loss)
            module_pred_losses.append(pred_loss)
            module_entropies.append(entropy)
            module_grad_norms.append(grad_norm)

        # --- 10. Update State ---
        self.ct = ct_plus_1

        # --- 11. Return Metrics ---
        finite_sm_log_cls = [s.item() for s in log_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_sm_raw_cls = [s.item() for s in raw_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_total_losses = [l for l in module_total_losses if l is not None and math.isfinite(l)]
        finite_policy_losses = [l for l in module_policy_losses if l is not None and math.isfinite(l)]
        finite_pred_losses = [l for l in module_pred_losses if l is not None and math.isfinite(l)]
        finite_entropies = [e for e in module_entropies if e is not None and math.isfinite(e)]
        finite_module_grads = [g for g in module_grad_norms if g is not None and math.isfinite(g)]

        # Get TaskHead specific surprises
        task_sm_log = self.task_head.last_log_surprise_sm_task.item() if self.task_head.last_log_surprise_sm_task is not None else float('nan')
        task_sm_raw = self.task_head.last_raw_surprise_sm_task.item() if self.task_head.last_raw_surprise_sm_task is not None else float('nan')

        metrics = {
            "Gt_log": global_log_surprise_gt if math.isfinite(global_log_surprise_gt) else float('nan'),
            "Gt_log_EMA": self.gt_log_ema,
            "Sm_log_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_log_std": np.std(finite_sm_log_cls) if len(finite_sm_log_cls) > 1 else 0.0,
            "Sm_log_cls_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_raw_cls_avg": np.mean(finite_sm_raw_cls) if finite_sm_raw_cls else float('nan'),
            "TaskHead_Sm_log_task": task_sm_log if math.isfinite(task_sm_log) else float('nan'),
            "TaskHead_Sm_raw_task": task_sm_raw if math.isfinite(task_sm_raw) else float('nan'),
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