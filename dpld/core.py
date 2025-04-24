# dpld/core.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math # For checking nan/inf
import traceback # For detailed error printing

from utils import estimate_lyapunov_exponent, sparsify_vector

# --- Constants ---
EPSILON = 1e-8
DEFAULT_ACTION_STD = 0.08
CLS_NORM_CLIP_VAL = 100.0
MODULE_OUTPUT_CLIP_VAL = 100.0
ACTION_MEAN_CLIP_VAL = 5.0
GRAD_CLIP_NORM = 1.0
META_REWARD_CLIP = 10.0
# DEFAULT_PREDICTION_LOSS_WEIGHT = 0.5 # No longer used by simplified TaskHead
# DEFAULT_TASK_LOSS_WEIGHT = 1.5 # No longer used by simplified TaskHead
TASKHEAD_POLICY_REWARD_SCALE = 1.0
MATH_ENCODER_SCALE = 0.2 # MODIFIED Rev 11: Further reduced encoder scale
# MODULE_WRITE_SCALE = 0.5 # Added Rev 11: Scale predictive module writes (Use cmd line arg)

# --- Math Encoder Module (MODIFIED Rev 11: Writes to fixed indices) ---
class MathEncoder(nn.Module):
    """
    Encodes an arithmetic problem (a, op, b) into a sparse CLS vector.
    MODIFIED Rev 11: Writes scaled values of a, op_idx, b to fixed indices 0, 1, 2.
    """
    def __init__(self, cls_dim, k_sparse_write, write_magnitude=1.0, device='cpu'): # Removed unused args
        super().__init__()
        self.cls_dim = cls_dim
        # self.k_sparse_write = k_sparse_write # k_sparse not used for fixed indices
        self.write_magnitude = write_magnitude
        self.device = device

    def forward(self, a, op_idx, b):
        a_val = a.float().item()
        op_val = op_idx.float().item()
        b_val = b.float().item()

        # Fixed indices for inputs
        indices = torch.tensor([[0, 1, 2]], dtype=torch.long, device=self.device)
        # Scale values before writing
        values = torch.tensor([a_val, op_val, b_val], dtype=torch.float32, device=self.device) * self.write_magnitude

        # Ensure indices are within bounds
        if torch.any(indices >= self.cls_dim):
            print(f"Warning: MathEncoder trying to write to index >= cls_dim ({self.cls_dim}). Clamping.")
            # This case shouldn't happen with fixed indices 0, 1, 2 if cls_dim is large enough
            values = values[indices < self.cls_dim]
            indices = indices[indices < self.cls_dim]
            if indices.numel() == 0: # No valid indices left
                 return torch.sparse_coo_tensor(indices.unsqueeze(0), values, (self.cls_dim,), device=self.device).coalesce()


        # Create sparse tensor directly
        i_math = torch.sparse_coo_tensor(indices, values, (self.cls_dim,), device=self.device)
        return i_math.coalesce()


# --- Predictive Module (Base Class - Unchanged from Rev 9) ---
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
                 prediction_loss_weight=1.0, # Default to 1.0 for base module CLS prediction
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
        self.prediction_loss_weight = prediction_loss_weight # Used for CLS prediction loss
        # self.task_loss_weight is unused here

        # fm: Predicts next CLS state
        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)

        # qm: Gating parameter for write action
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01)

        self.register_buffer('sm_log_baseline', torch.tensor(0.0, device=device))
        self.last_prediction_ct_plus_1 = None # Stores grad-enabled CLS prediction
        self.last_log_surprise_sm = None # Grad-enabled combined loss value (only CLS for base)
        self.last_log_surprise_sm_cls = torch.tensor(0.0, device=device) # Detached CLS log surprise
        self.last_raw_surprise_sm_cls = None # Detached raw CLS surprise
        self.last_log_prob = None # Log prob of the write action taken
        self.last_action_dist = None # Distribution used for action sampling

        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate, weight_decay=weight_decay)

    def predict(self, ct):
        """Predicts the next CLS state."""
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        # Use detached input for prediction to avoid gradients flowing back through ct
        prediction = self.fm(ct_dense.detach())
        # Clamp output magnitude for stability
        prediction = torch.clamp(prediction, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_prediction_ct_plus_1 = prediction # Store grad-enabled prediction
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates local LOG-SURPRISE Sm (only CLS part for base module)."""
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             # Initialize if no prediction exists
             self.last_log_surprise_sm = torch.tensor(0.0, device=self.device)
             self.last_log_surprise_sm_cls = self.sm_log_baseline.detach()
             self.last_raw_surprise_sm_cls = torch.tensor(float('nan'), device=self.device)
             return self.last_log_surprise_sm

        # Check for non-finite prediction
        if not torch.all(torch.isfinite(self.last_prediction_ct_plus_1)):
            print(f"Warning: Non-finite prediction in {self.__class__.__name__}.calculate_surprise")
            sm_cls_raw = torch.tensor(1e6, device=self.device)
            # Ensure it has grad if needed, though it shouldn't be used for backprop if prediction was bad
            if torch.is_grad_enabled(): sm_cls_raw = sm_cls_raw.requires_grad_()
        else:
            # Calculate MSE loss between prediction (with grad) and actual next state (detached)
            sm_cls_raw = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')

        # Store detached raw surprise for logging
        self.last_raw_surprise_sm_cls = sm_cls_raw.detach()

        # Calculate log-surprise (use raw surprise with grad history)
        sm_cls_log = torch.log1p(sm_cls_raw + EPSILON)

        # Store detached log surprise for influence scaling and logging
        self.last_log_surprise_sm_cls = sm_cls_log.detach()

        # Combined surprise for loss = weighted CLS surprise (with grad history)
        combined_log_surprise = self.prediction_loss_weight * sm_cls_log

        # Handle potential NaN/Inf in combined surprise
        if not math.isfinite(combined_log_surprise.item()):
             print(f"Warning: Non-finite combined surprise in {self.__class__.__name__}.calculate_surprise")
             # Use baseline value, ensure no grad history to prevent backprop issues
             self.last_log_surprise_sm = torch.tensor(self.sm_log_baseline.item() + 1.0, device=self.device)
        else:
            self.last_log_surprise_sm = combined_log_surprise # Store grad-enabled combined surprise for loss

            # Update EMA baseline using detached CLS log surprise
            with torch.no_grad():
                update_val = self.last_log_surprise_sm_cls if torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline
                self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                       (1 - self.surprise_baseline_ema) * update_val

        return self.last_log_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse write vector Im using LOG-SURPRISE (CLS part) for alpha_m."""
        # Use detached prediction as the basis for the write vector (vm)
        if self.last_prediction_ct_plus_1 is None:
             vm = torch.zeros(self.cls_dim, device=self.device)
        else:
             vm = self.last_prediction_ct_plus_1.detach()

        # Use detached CLS log surprise for influence calculation
        current_log_surprise_detached = self.last_log_surprise_sm_cls if self.last_log_surprise_sm_cls is not None and torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline.detach()
        log_surprise_diff = current_log_surprise_detached - self.sm_log_baseline.detach()

        # Calculate gating based on current CLS state (detached) and qm (parameter)
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        tau_g = 1.0
        # Ensure ct_dense is detached for gating calculation to isolate qm gradient
        gate_raw_score = self.qm * ct_dense.detach()
        gate_activation_gm = torch.sigmoid(gate_raw_score / tau_g) # Grad flows back to qm

        # Calculate influence scalar (detached)
        alpha_base = 1.0; alpha_scale = 1.0
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * log_surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0).detach() # Detach alpha

        # Calculate action mean (grad flows back to qm via gate_activation_gm)
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm) # vm is detached
        action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

        # Handle potential non-finite values in action mean
        if not torch.all(torch.isfinite(action_mean)):
            print(f"Warning: Non-finite action_mean in {self.__class__.__name__}.generate_write_vector")
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=ACTION_MEAN_CLIP_VAL, neginf=-ACTION_MEAN_CLIP_VAL)

        # Sample action from Normal distribution centered at action_mean
        action_std_tensor = torch.full_like(action_mean, fill_value=self.action_std)
        try:
            dist = Normal(action_mean, action_std_tensor) # action_mean has grad history to qm
            self.last_action_dist = dist
            dense_write_vector_sampled = dist.rsample() # Use rsample() for reparameterization trick if needed, though sample() is fine for REINFORCE
            # Calculate log_prob using the sampled action (detached)
            self.last_log_prob = dist.log_prob(dense_write_vector_sampled.detach()).sum()
        except ValueError as e:
             print(f"ERROR creating Normal distribution in {self.__class__.__name__}: {e}")
             dense_write_vector_sampled = torch.zeros_like(action_mean)
             self.last_log_prob = None
             self.last_action_dist = None

        # Sparsify the sampled action (detached for the final output)
        write_vector_im_sparse_vals = sparsify_vector(dense_write_vector_sampled.detach(), self.k_sparse_write)
        sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]

        # Create the final sparse tensor output
        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=dense_write_vector_sampled.dtype)
        else:
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=dense_write_vector_sampled.dtype, device=self.device),
                                                 (self.cls_dim,))
        return im_sparse.coalesce()

    def learn(self, reward_signal_detached, step): # Renamed reward signal input
        """
        Updates base module parameters (fm and qm) using combined loss.
        Policy reward uses the provided detached reward signal.
        Prediction loss uses weighted CLS surprise calculated earlier.
        """
        policy_loss_item = 0.0
        prediction_loss_item = 0.0
        entropy_item = 0.0
        grad_norm = 0.0

        # --- 1. Calculate Policy Loss Component (for qm) ---
        policy_loss = torch.tensor(0.0, device=self.device)
        if self.last_log_prob is not None and self.last_action_dist is not None and \
           reward_signal_detached is not None and math.isfinite(reward_signal_detached.item()):

            reward_tensor = reward_signal_detached # Use the provided reward

            entropy_tensor = torch.tensor(0.0, device=self.device)
            try:
                if self.last_action_dist is not None:
                    entropy_tensor = self.last_action_dist.entropy().sum()
                    if not torch.isfinite(entropy_tensor): entropy_tensor = torch.tensor(0.0, device=self.device)
                entropy_item = entropy_tensor.item()

            except Exception as e:
                 print(f"Warning: Entropy calculation error in PredictiveModule: {e}")
                 entropy_tensor = torch.tensor(0.0, device=self.device); entropy_item = 0.0

            entropy_term = self.entropy_coeff * entropy_tensor

            # REINFORCE: -(reward * log_prob + entropy_bonus)
            if torch.isfinite(self.last_log_prob):
                policy_loss = -(reward_tensor * self.last_log_prob + entropy_term)
                if not torch.isfinite(policy_loss):
                    print(f"Warning: Non-finite policy loss component in {self.__class__.__name__}")
                    policy_loss = torch.tensor(0.0, device=self.device)
            else:
                 print(f"Warning: Non-finite log_prob in {self.__class__.__name__}")
                 policy_loss = torch.tensor(0.0, device=self.device)
        policy_loss_item = policy_loss.item()

        # --- 2. Calculate Prediction Loss Component (for fm) ---
        prediction_loss = torch.tensor(0.0, device=self.device)
        # self.last_log_surprise_sm has grad history back to fm (set in calculate_surprise)
        if self.last_log_surprise_sm is not None and torch.is_tensor(self.last_log_surprise_sm) and self.last_log_surprise_sm.requires_grad:
            if torch.isfinite(self.last_log_surprise_sm):
                prediction_loss = self.last_log_surprise_sm # Already weighted in calculate_surprise
            else:
                print(f"Warning: Non-finite prediction loss component (last_log_surprise_sm) in {self.__class__.__name__}")
                prediction_loss = torch.tensor(0.0, device=self.device)
        prediction_loss_item = prediction_loss.item()

        # --- 3. Combine Losses ---
        total_loss = policy_loss + prediction_loss

        # --- 4. Backpropagation and Optimizer Step ---
        if torch.isfinite(total_loss) and total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients for fm parameters and qm
            params_to_clip = list(self.fm.parameters()) + [self.qm]

            # Calculate grad norm before clipping for logging
            total_norm_sq = 0.0
            for p in params_to_clip:
                if p.grad is not None:
                     param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                     if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
            grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

            # Apply gradient clipping
            nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)

            self.optimizer.step()
        else:
            grad_norm = 0.0 # No update happened
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN/Inf total_loss in {self.__class__.__name__}.learn(). Skipping step. Policy: {policy_loss_item}, Pred: {prediction_loss_item}")

        # Clear stored values used for learning
        self.last_log_prob = None
        self.last_action_dist = None
        # Keep last_log_surprise_sm_cls for next step's influence scaling

        return total_loss.item(), policy_loss_item, prediction_loss_item, entropy_item, grad_norm


# --- Task Head Module (MODIFIED Rev 11: Simplified MLP, fixed read/write) ---
class TaskHead(nn.Module): # MODIFIED Rev 11: Inherit directly from nn.Module
    """
    Simplified TaskHead for arithmetic task (Revision 11).
    Reads fixed CLS indices [0, 1, 2] for inputs (a, op, b).
    Uses an MLP to predict the answer 'c'.
    Writes the predicted 'c' to fixed CLS index 3.
    Loss is *only* MSE on the task prediction.
    No policy learning (qm) or CLS prediction.
    Still calculates detached CLS surprise for influence scaling in generate_write_vector.
    """
    def __init__(self, cls_dim, module_hidden_dim, taskhead_lr,
                 task_loss_weight=1.0, # Weight for Task part
                 weight_decay=1e-5, device='cpu',
                 # Added base class args for compatibility in DPLDSystem init list
                 k_sparse_write=0.0, entropy_coeff=0.0, surprise_scale_factor=1.0,
                 surprise_baseline_ema=0.99, action_std=0.0, prediction_loss_weight=0.0):
        super().__init__()
        self.cls_dim = cls_dim
        self.device = device
        self.task_loss_weight = task_loss_weight
        self.write_index = 3 # Fixed index to write the predicted answer

        # --- fm: Prediction Network (Simple MLP) ---
        # Input size is 3 (a, op, b from fixed indices)
        self.fm = nn.Sequential(
            nn.Linear(3, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(module_hidden_dim // 2, 1) # Output is scalar prediction 'c'
        ).to(device)

        # Optimizer for the prediction network (fm)
        self.optimizer = optim.Adam(self.fm.parameters(), lr=taskhead_lr, weight_decay=weight_decay)

        # --- Attributes needed for compatibility / logging ---
        self.last_task_prediction = None # Stores grad-enabled task prediction
        self.last_log_surprise_sm = None # Grad-enabled loss value (only task for this module)
        self.last_log_surprise_sm_task = torch.tensor(0.0, device=self.device) # Detached task log surprise
        self.last_raw_surprise_sm_task = None # Detached raw task surprise

        # --- Attributes needed for influence scaling in generate_write_vector ---
        # These are calculated but don't contribute to prediction loss
        self.last_prediction_ct_plus_1_dummy = None # Dummy CLS prediction
        self.register_buffer('sm_log_baseline', torch.tensor(0.0, device=device)) # Baseline for CLS surprise
        self.last_log_surprise_sm_cls = torch.tensor(0.0, device=device) # Detached CLS log surprise
        self.last_raw_surprise_sm_cls = None # Detached raw CLS surprise
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.01) # Keep qm for influence calc compatibility
        self.k_sparse_write = 0.01 # Use a small default for influence calc compatibility
        self.surprise_scale_factor = surprise_scale_factor
        self.surprise_baseline_ema = surprise_baseline_ema


    def predict(self, ct):
        """Reads fixed indices [0, 1, 2] and predicts task answer 'c'."""
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        # Read fixed input indices
        # Add checks for dimension and index bounds
        if ct_dense.dim() == 0 or ct_dense.numel() < 3:
             print("Warning: CLS state too small in TaskHead.predict")
             # Return a dummy prediction or handle error appropriately
             dummy_pred = torch.tensor(0.0, device=self.device)
             self.last_task_prediction = dummy_pred
             self.last_prediction_ct_plus_1_dummy = torch.zeros(self.cls_dim, device=self.device) # Dummy CLS pred
             return self.last_prediction_ct_plus_1_dummy # Return dummy CLS prediction

        a_in = ct_dense[0].detach()
        op_in = ct_dense[1].detach()
        b_in = ct_dense[2].detach()
        mlp_input = torch.stack([a_in, op_in, b_in])

        # Predict task answer using MLP
        predicted_answer = self.fm(mlp_input).squeeze() # Squeeze to get scalar
        # Clamp output magnitude
        predicted_answer_clamped = torch.clamp(predicted_answer, -MODULE_OUTPUT_CLIP_VAL, MODULE_OUTPUT_CLIP_VAL)
        self.last_task_prediction = predicted_answer_clamped # Store grad-enabled prediction

        # Create a dummy CLS prediction (e.g., zeros) as this module doesn't predict CLS
        self.last_prediction_ct_plus_1_dummy = torch.zeros(self.cls_dim, device=self.device)

        # Return the dummy CLS prediction for compatibility with base class usage
        return self.last_prediction_ct_plus_1_dummy

    def calculate_surprise(self, actual_ct_plus_1, true_task_output=None):
        """Calculates LOG-SURPRISE Sm (only task part for loss)."""

        # --- Calculate Task Surprise (for loss) ---
        if true_task_output is not None and self.last_task_prediction is not None and \
           torch.isfinite(self.last_task_prediction) and torch.isfinite(true_task_output):
            # Calculate MSE loss between prediction (with grad) and true answer (detached)
            sm_task_raw = F.mse_loss(self.last_task_prediction, true_task_output.float().detach(), reduction='mean')
        else:
            # Handle cases where prediction or true output is invalid
            print(f"Warning: Invalid inputs for TaskHead surprise calculation. Pred: {self.last_task_prediction}, True: {true_task_output}")
            sm_task_raw = torch.tensor(1e6, device=self.device)
            # Ensure it has grad if needed, though it shouldn't be used for backprop if inputs were bad
            if torch.is_grad_enabled(): sm_task_raw = sm_task_raw.requires_grad_()

        # Store detached raw task surprise for logging
        self.last_raw_surprise_sm_task = sm_task_raw.detach()
        # Calculate log-surprise (use raw surprise with grad history)
        sm_task_log = torch.log1p(sm_task_raw + EPSILON)
        # Store detached log task surprise for logging
        self.last_log_surprise_sm_task = sm_task_log.detach()

        # --- Combined surprise for loss = weighted TASK surprise ONLY ---
        combined_log_surprise = self.task_loss_weight * sm_task_log

        # Handle potential NaN/Inf in combined surprise
        if not math.isfinite(combined_log_surprise.item()):
             print(f"Warning: Non-finite combined surprise in TaskHead.calculate_surprise")
             # Use a default high value, ensure no grad history
             self.last_log_surprise_sm = torch.tensor(10.0, device=self.device) # Arbitrary high log surprise
        else:
            self.last_log_surprise_sm = combined_log_surprise # Store grad-enabled task surprise for loss

        # --- Calculate Dummy CLS Surprise (for influence scaling compatibility) ---
        # This part does NOT contribute to the gradient/loss for TaskHead's fm
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1
        dummy_cls_pred = self.last_prediction_ct_plus_1_dummy.detach() # Use detached dummy pred
        sm_cls_raw_dummy = F.mse_loss(dummy_cls_pred, actual_ct_plus_1_dense.detach(), reduction='mean')
        self.last_raw_surprise_sm_cls = sm_cls_raw_dummy.detach()
        sm_cls_log_dummy = torch.log1p(sm_cls_raw_dummy + EPSILON)
        self.last_log_surprise_sm_cls = sm_cls_log_dummy.detach() # Store detached dummy CLS log surprise

        # Update EMA baseline using the dummy CLS surprise
        with torch.no_grad():
            update_val = self.last_log_surprise_sm_cls if torch.isfinite(self.last_log_surprise_sm_cls) else self.sm_log_baseline
            self.sm_log_baseline = self.surprise_baseline_ema * self.sm_log_baseline + \
                                   (1 - self.surprise_baseline_ema) * update_val

        # Return the grad-enabled combined (task-only) surprise for loss calculation
        return self.last_log_surprise_sm

    def generate_write_vector(self, ct):
        """Generates a sparse write vector writing the predicted 'c' to index 3."""
        if self.last_task_prediction is None:
            predicted_c = torch.tensor(0.0, device=self.device)
        else:
            # Use detached prediction for the write value
            predicted_c = self.last_task_prediction.detach()

        # Ensure the value is finite
        if not torch.isfinite(predicted_c):
            predicted_c = torch.tensor(0.0, device=self.device)

        # Fixed index and value for the write
        indices = torch.tensor([[self.write_index]], dtype=torch.long, device=self.device)
        values = torch.tensor([predicted_c.item()], dtype=torch.float32, device=self.device)

        # Create sparse tensor
        if indices.numel() > 0 and indices[0,0] < self.cls_dim:
             im_sparse = torch.sparse_coo_tensor(indices, values, (self.cls_dim,), device=self.device)
        else:
             # Handle case where index is out of bounds (shouldn't happen with fixed index 3 if cls_dim >= 4)
             print(f"Warning: TaskHead write index {self.write_index} out of bounds for cls_dim {self.cls_dim}")
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=torch.float32, device=self.device),
                                                 (self.cls_dim,))
        return im_sparse.coalesce()

    def learn(self, reward_signal_detached, step): # Reward signal is ignored now
        """Updates TaskHead MLP parameters (fm) using only prediction loss."""
        policy_loss_item = 0.0 # No policy loss
        prediction_loss_item = 0.0
        entropy_item = 0.0 # No policy/entropy
        grad_norm = 0.0

        # --- Prediction Loss Component (for fm=MLP) ---
        prediction_loss = torch.tensor(0.0, device=self.device)
        # self.last_log_surprise_sm has grad history back to fm (set in calculate_surprise)
        # and now only contains the weighted task surprise.
        if self.last_log_surprise_sm is not None and torch.is_tensor(self.last_log_surprise_sm) and self.last_log_surprise_sm.requires_grad:
            if torch.isfinite(self.last_log_surprise_sm):
                prediction_loss = self.last_log_surprise_sm # This is the weighted task loss
            else:
                print(f"Warning: Non-finite prediction loss component in TaskHead.learn")
                prediction_loss = torch.tensor(0.0, device=self.device)
        prediction_loss_item = prediction_loss.item()

        # --- Total Loss = Prediction Loss Only ---
        total_loss = prediction_loss

        # --- Backpropagation and Optimizer Step ---
        if torch.isfinite(total_loss) and total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients for fm parameters
            params_to_clip = list(self.fm.parameters())

            # Calculate grad norm before clipping
            total_norm_sq = 0.0
            for p in params_to_clip:
                if p.grad is not None:
                     param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                     if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
            grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')

            # Apply gradient clipping
            nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)

            self.optimizer.step()
        else:
            grad_norm = 0.0 # No update
            if not torch.isfinite(total_loss):
                print(f"Warning: NaN/Inf total_loss in TaskHead.learn(). Skipping step. PredLoss: {prediction_loss_item}")

        # No policy-related values to clear

        return total_loss.item(), policy_loss_item, prediction_loss_item, entropy_item, grad_norm

    def get_task_prediction(self):
        """Returns the detached scalar task prediction."""
        # Ensure it returns detached value
        return self.last_task_prediction.detach() if self.last_task_prediction is not None else None


# --- Meta-Model (Unchanged from Rev 9) ---
# ... (keep MetaModel class as is) ...
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
        # Use stability target if LE is invalid
        lambda_val = smoothed_lambda_max if smoothed_lambda_max is not None and math.isfinite(smoothed_lambda_max) else self.stability_target
        gt_val = smoothed_gt if smoothed_gt is not None and math.isfinite(smoothed_gt) else 0.0 # Use 0 if Gt is invalid

        meta_input_base = torch.tensor([gt_val, lambda_val], dtype=torch.float32, device=self.device)
        # Reshape for GRU: (batch, seq, feature) -> (1, 1, 2)
        meta_input = meta_input_base.unsqueeze(0).unsqueeze(0)
        self.last_meta_input_smoothed = meta_input.detach().clone() # Store for learning

        # Handle hidden state shape mismatch if batch size changes (shouldn't here)
        hidden_state_input = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        if hidden_state_input is not None and hidden_state_input.shape[1] != meta_input.shape[0]:
             print("Warning: MetaModel hidden state batch dimension mismatch. Resetting.")
             hidden_state_input = None

        # Compute GRU output without tracking gradients for regulation decision
        with torch.no_grad():
            gru_output, next_hidden_state = self.gru(meta_input, hidden_state_input)
            self.last_gru_hidden_state = next_hidden_state.detach() # Store for next step
            last_hidden = gru_output[:, -1, :] # Get last time step output
            gamma_offset_signal = self.output_layer(last_hidden).squeeze() # Get scalar output

        # Store the raw signal for learning (used as action mean)
        self.last_gamma_offset_signal = gamma_offset_signal.detach().clone()

        # Transform signal to gamma_t value
        gamma_offset_tanh = torch.tanh(gamma_offset_signal) # Squash to [-1, 1]
        gamma_center = (self.gamma_max + self.gamma_min) / 2.0
        gamma_range = (self.gamma_max - self.gamma_min) / 2.0
        gamma_t = gamma_center + gamma_range * gamma_offset_tanh
        gamma_t = torch.clamp(gamma_t, self.gamma_min, self.gamma_max) # Ensure bounds

        # mmod_t is currently zero
        mmod_t = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                         torch.empty((0,), dtype=torch.float32, device=self.device),
                                         (self.cls_dim,))
        return gamma_t.item(), mmod_t

    def learn(self, Gt_log_next, lambda_max_next):
        """Updates Meta-Model parameters using clipped reward based on log-surprise and instability."""
        grad_norm = 0.0
        # Check if we have stored state from the previous compute_regulation call
        if self.last_meta_input_smoothed is None or self.last_gamma_offset_signal is None:
            return 0.0, grad_norm # Cannot learn without previous action/state

        # Validate inputs for reward calculation
        if Gt_log_next is None or not math.isfinite(Gt_log_next):
             print("Warning: Invalid Gt_log_next in MetaModel.learn. Skipping update.")
             return 0.0, grad_norm
        # Use stability target if LE is invalid
        lambda_max_reward = lambda_max_next if lambda_max_next is not None and math.isfinite(lambda_max_next) else self.stability_target

        # Calculate reward components
        surprise_term = self.surprise_weight * torch.tensor(Gt_log_next, device=self.device)
        instability_term = self.instability_weight * F.relu(torch.tensor(lambda_max_reward - self.stability_target, device=self.device))
        # Reward is negative of cost (we want to minimize surprise and instability)
        raw_reward = -(surprise_term + instability_term).detach()
        # Clip reward for stability
        clipped_reward = torch.clamp(raw_reward, -META_REWARD_CLIP, META_REWARD_CLIP)

        # --- Policy Gradient Calculation ---
        # Recompute GRU output with gradients enabled using stored input
        hidden_state_input_rerun = self.last_gru_hidden_state.detach() if self.last_gru_hidden_state is not None else None
        if hidden_state_input_rerun is not None and hidden_state_input_rerun.shape[1] != self.last_meta_input_smoothed.shape[0]:
             hidden_state_input_rerun = None # Reset if shape mismatch

        gru_output_rerun, _ = self.gru(self.last_meta_input_smoothed, hidden_state_input_rerun)
        last_hidden_rerun = gru_output_rerun[:, -1, :]
        # Recompute the action mean (gamma_offset_signal) with gradients
        gamma_offset_signal_mean = self.output_layer(last_hidden_rerun).squeeze()

        # Assume a fixed standard deviation for the meta-action distribution
        meta_action_std = torch.tensor(0.1, device=self.device) # Fixed exploration
        try:
            # Create distribution centered at the recomputed mean
            meta_action_dist = Normal(gamma_offset_signal_mean, meta_action_std)
            # Calculate log_prob of the *actual action taken* (stored signal)
            log_prob_meta_action = meta_action_dist.log_prob(self.last_gamma_offset_signal)
        except ValueError as e:
             print(f"ERROR creating Meta Normal distribution: {e}")
             self._clear_last_state() # Clear state if distribution failed
             return 0.0, grad_norm

        # Check if log_prob is valid
        if not torch.isfinite(log_prob_meta_action):
             print("Warning: Non-finite log_prob_meta_action in MetaModel.learn. Skipping update.")
             self._clear_last_state()
             return 0.0, grad_norm

        # Policy gradient loss: - (reward * log_prob)
        policy_gradient_loss = -clipped_reward * log_prob_meta_action

        if not torch.isfinite(policy_gradient_loss):
             print("Warning: Non-finite policy_gradient_loss in MetaModel.learn. Skipping update.")
             self._clear_last_state()
             return 0.0, grad_norm

        # --- Backpropagation and Optimizer Step ---
        self.optimizer.zero_grad()
        policy_gradient_loss.backward()

        # Clip gradients
        params_to_clip = list(self.gru.parameters()) + list(self.output_layer.parameters())
        total_norm_sq = 0.0
        for p in params_to_clip:
            if p.grad is not None:
                param_norm_sq = p.grad.detach().data.norm(2).pow(2)
                if math.isfinite(param_norm_sq): total_norm_sq += param_norm_sq
        grad_norm = math.sqrt(total_norm_sq) if math.isfinite(total_norm_sq) else float('nan')
        nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)

        self.optimizer.step()

        # Clear stored state after learning step
        self._clear_last_state()

        return policy_gradient_loss.item(), grad_norm

    def _clear_last_state(self):
        """Clears stored state used for learning."""
        self.last_meta_input_smoothed = None
        self.last_gamma_offset_signal = None


# --- DPLD System (MODIFIED Rev 11: Module write scaling, LE dynamics map adjusted) ---
class DPLDSystem(nn.Module):
    """
    Main DPLD system coordinating CLS, Modules, Meta-Model, and Task components.
    MODIFIED Rev 11: Simplified TaskHead, added module write scaling.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, taskhead_lr, noise_std_dev_schedule,
                 env, embedding_dim, # embedding_dim now unused by MathEncoder
                 entropy_coeff=0.01,
                 ema_alpha=0.99,
                 gamma_min=0.01, gamma_max=0.2, stability_target=-0.01,
                 action_std=DEFAULT_ACTION_STD,
                 # prediction_loss_weight=0.5, # No longer used by TaskHead
                 # task_loss_weight=1.5, # No longer used by TaskHead
                 module_write_scale=1.0, # Added Rev 11
                 weight_decay=1e-5,
                 meta_input_dim=2, clip_cls_norm=True, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules
        self.k_sparse_write = k_sparse_write # Used by generic modules
        self.noise_std_dev_schedule = noise_std_dev_schedule
        self.clip_cls_norm = clip_cls_norm
        self.device = device
        self.env = env
        self.ema_alpha = ema_alpha
        self.module_write_scale = module_write_scale # Added Rev 11

        self.ct = self._init_cls()

        # Math encoder now simpler
        self.math_encoder = MathEncoder(cls_dim, k_sparse_write, device=device).to(device)

        # Task head now simpler, uses taskhead_lr
        self.task_head = TaskHead(cls_dim, module_hidden_dim, taskhead_lr,
                                  task_loss_weight=1.0, # Task loss is the only prediction loss
                                  weight_decay=weight_decay, device=device).to(device)

        # Generic predictive modules use module_lr
        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr,
                             entropy_coeff=entropy_coeff,
                             action_std=action_std,
                             prediction_loss_weight=1.0, # Base modules predict CLS
                             task_loss_weight=0.0,
                             weight_decay=weight_decay, device=device)
            for _ in range(num_modules)
        ])

        # Meta model uses meta_lr
        self.meta_model = MetaModel(meta_input_dim, meta_hidden_dim, meta_lr, cls_dim,
                                    gamma_min, gamma_max, stability_target,
                                    weight_decay=weight_decay, device=device).to(device)

        # EMA buffers
        self.gt_log_ema = None
        self.lambda_max_ema = None

    def _init_cls(self):
        """Initializes CLS state."""
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.01
        # Ensure initial state is sparse but respects k_sparse conceptually
        sparse_ct = sparsify_vector(initial_dense, 0.1).to_sparse_coo() # Use a fixed initial sparsity
        return sparse_ct.coalesce()

    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation (Unchanged)."""
        # Ensure inputs are sparse
        ct = ct if ct.is_sparse else ct.to_sparse_coo()
        sum_im = sum_im if sum_im.is_sparse else sum_im.to_sparse_coo()
        mmodt = mmodt if mmodt.is_sparse else mmodt.to_sparse_coo()

        # Check for non-finite values before proceeding
        if not torch.all(torch.isfinite(ct.values())): ct = self._init_cls()
        if not torch.all(torch.isfinite(sum_im.values())): sum_im = self._init_cls() # Or zero sparse tensor
        if not torch.all(torch.isfinite(mmodt.values())): mmodt = self._init_cls() # Or zero sparse tensor

        # Perform update
        decayed_ct = (1.0 - gamma_t) * ct
        combined_inputs = (decayed_ct + sum_im + mmodt).coalesce()

        # Add noise
        noise_et = torch.randn(self.cls_dim, device=self.device) * noise_std_dev
        ct_plus_1_dense = combined_inputs.to_dense() + noise_et

        # Check and handle non-finite values after noise addition
        if not torch.all(torch.isfinite(ct_plus_1_dense)):
            print("Warning: Non-finite state after noise addition. Clamping.")
            ct_plus_1_dense = torch.nan_to_num(ct_plus_1_dense, nan=0.0, posinf=CLS_NORM_CLIP_VAL, neginf=-CLS_NORM_CLIP_VAL)

        # Apply norm clipping
        if self.clip_cls_norm:
            norm = torch.linalg.norm(ct_plus_1_dense)
            if norm > CLS_NORM_CLIP_VAL:
                 ct_plus_1_dense = ct_plus_1_dense * (CLS_NORM_CLIP_VAL / (norm + EPSILON))

        # Sparsify the final dense state
        # Use k_sparse_write from generic modules for overall sparsity target
        ct_plus_1_sparse_vals = sparsify_vector(ct_plus_1_dense, self.k_sparse_write)
        sparse_indices = torch.where(ct_plus_1_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = ct_plus_1_sparse_vals[sparse_indices.squeeze(0)]

        # Create the final sparse tensor
        if sparse_indices.numel() > 0:
             ct_plus_1 = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device, dtype=ct_plus_1_dense.dtype)
        else:
             # Handle case of all zeros after sparsification
             ct_plus_1 = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=ct_plus_1_dense.dtype, device=self.device),
                                                 (self.cls_dim,))
        return ct_plus_1.coalesce()


    def get_dynamics_map(self, fixed_gamma, fixed_noise_std=0.0):
         """
         Returns a function for LE estimation.
         MODIFIED Rev 11: Excludes simplified TaskHead write from dynamics.
         Uses generic module k_sparse_write for final sparsification.
         Ensures output is float64.
         """
         gamma_val = float(fixed_gamma)
         # Only consider generic predictive modules for LE dynamics
         modules_for_le = self.pred_modules
         k_sparse_le = self.k_sparse_write # Use system's k_sparse

         def dynamics_map(state_t_dense):
             # Ensure input is float64 and requires grad
             state_t_dense = state_t_dense.clone().to(dtype=torch.float64).requires_grad_(True)

             if not torch.all(torch.isfinite(state_t_dense)):
                 print("Warning (LE Dynamics Map): Input state non-finite.")
                 # Return detached zero tensor of correct dtype/device
                 return torch.zeros_like(state_t_dense, requires_grad=False).detach()

             state_t_sparse = state_t_dense.detach().to_sparse_coo()

             # Calculate sum_im from generic modules only, without grad for this part
             sum_im_le = torch.zeros(self.cls_dim, device=self.device, dtype=torch.float64)
             with torch.no_grad():
                 for module in modules_for_le:
                     # Predict (returns dummy CLS pred for TaskHead, actual for generic)
                     pred_cls_detached = module.predict(state_t_sparse).detach().to(dtype=torch.float64)
                     vm = pred_cls_detached

                     # Calculate influence based on detached state and baseline
                     current_log_surprise_detached = module.sm_log_baseline.detach() # Baseline is float32, cast if needed
                     log_surprise_diff = current_log_surprise_detached - module.sm_log_baseline.detach()
                     # Use detached state for gating score calculation
                     gate_raw_score = module.qm.detach() * state_t_dense.detach().to(module.qm.dtype) # Match qm dtype
                     gate_activation_gm = torch.sigmoid(gate_raw_score / 1.0)
                     influence_scalar_am = 1.0 + 1.0 * torch.tanh(module.surprise_scale_factor * log_surprise_diff)
                     influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)

                     # Calculate action mean (deterministic part of write)
                     intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
                     action_mean = torch.clamp(intermediate_write_vector, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

                     # Sparsify the deterministic mean write
                     write_vector_im_sparse_vals = sparsify_vector(action_mean, module.k_sparse_write)
                     sum_im_le += write_vector_im_sparse_vals # Add sparse contribution

             # --- Calculate next state WITH gradient tracking ---
             # Decay term (tracks grad to input state_t_dense)
             decayed_ct_grad = (1.0 - gamma_val) * state_t_dense

             # Gating-dependent sum_im part (tracks grad to input state_t_dense via gating)
             gating_dependent_sum_im = torch.zeros_like(state_t_dense, dtype=torch.float64)
             # Need module predictions again, potentially detached
             with torch.no_grad():
                 module_preds = [m.predict(state_t_sparse).detach().to(dtype=torch.float64) for m in modules_for_le]

             for i, module in enumerate(modules_for_le):
                 vm = module_preds[i] # Detached prediction
                 # Gating depends on state_t_dense (which requires grad) and qm (detached for LE)
                 gate_raw_score_grad = module.qm.detach() * state_t_dense.to(module.qm.dtype) # Use input state with grad
                 gate_activation_gm_grad = torch.sigmoid(gate_raw_score_grad / 1.0)

                 # Influence scalar is detached
                 current_log_surprise_detached = module.sm_log_baseline.detach()
                 log_surprise_diff = current_log_surprise_detached - module.sm_log_baseline.detach()
                 influence_scalar_am = 1.0 + 1.0 * torch.tanh(module.surprise_scale_factor * log_surprise_diff)
                 influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)

                 # Calculate action mean with grad flowing from gating
                 intermediate_write_vector_grad = influence_scalar_am * (gate_activation_gm_grad * vm)
                 action_mean_grad = torch.clamp(intermediate_write_vector_grad, -ACTION_MEAN_CLIP_VAL, ACTION_MEAN_CLIP_VAL)

                 # Sparsify (this operation is non-differentiable, but grad flows through values)
                 write_vals_grad = sparsify_vector(action_mean_grad, module.k_sparse_write)
                 gating_dependent_sum_im += write_vals_grad.to(dtype=torch.float64)

             # Add noise (detached)
             noise_et = torch.randn(self.cls_dim, device=self.device, dtype=torch.float64) * fixed_noise_std

             # Combine terms
             next_state_dense = decayed_ct_grad + gating_dependent_sum_im + noise_et

             # --- Final Checks and Output ---
             if not torch.all(torch.isfinite(next_state_dense)):
                 print("Warning (LE Dynamics Map): Output state non-finite before clipping.")
                 # Return detached zero tensor if non-finite
                 return torch.zeros_like(state_t_dense, requires_grad=False).detach()

             # Apply norm clipping (without grad for LE)
             if self.clip_cls_norm:
                 with torch.no_grad():
                     norm = torch.linalg.norm(next_state_dense)
                     if norm > CLS_NORM_CLIP_VAL:
                         next_state_dense_clipped = next_state_dense * (CLS_NORM_CLIP_VAL / (norm + EPSILON))
                     else:
                         next_state_dense_clipped = next_state_dense
             else:
                 next_state_dense_clipped = next_state_dense

             # Return the dense state (float64) with gradients attached
             return next_state_dense_clipped

         return dynamics_map


    def step(self, current_step_num, task_input, estimate_le=False, true_answer_c=None):
        """
        Performs one full step.
        MODIFIED Rev 11: Simplified TaskHead logic, added module write scaling.
        """
        a, op_idx, b, _ = task_input # True answer c is passed separately

        # --- Safety Check: Reset CLS if non-finite ---
        if not torch.all(torch.isfinite(self.ct.values())):
             print(f"CRITICAL WARNING: CLS non-finite at start of step {current_step_num}. Resetting.")
             self.ct = self._init_cls()
             # Reset EMA buffers and module baselines? Maybe not necessary if CLS reset fixes it.
             self.gt_log_ema = None; self.lambda_max_ema = None

        ct_prev_dense = self.ct.to_dense().detach() # For LE estimation

        # --- 1. Module Predictions ---
        # Includes simplified TaskHead.predict()
        all_modules_inc_taskhead = self.pred_modules + [self.task_head]
        for module in all_modules_inc_taskhead:
            _ = module.predict(self.ct) # TaskHead predicts task answer, stores dummy CLS pred

        # --- 2. Meta-Model Regulation ---
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.gt_log_ema, self.lambda_max_ema)

        # --- 3. Module Write Vectors ---
        sum_im = self._init_cls() # Start with zero sparse tensor

        # Math encoder write (fixed indices)
        i_math = self.math_encoder(a, op_idx, b)
        sum_im += i_math

        # Generic predictive module writes (scaled)
        for module in self.pred_modules:
             im = module.generate_write_vector(self.ct)
             # Apply scaling to generic module writes
             sum_im += im * self.module_write_scale

        # TaskHead write (fixed index, predicted value)
        im_task = self.task_head.generate_write_vector(self.ct)
        sum_im += im_task

        sum_im = sum_im.coalesce() # Combine all sparse writes

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual LOG-Surprises ---
        log_surprises_sm_cls_detached = [] # For generic modules + taskhead dummy
        raw_surprises_sm_cls_detached = []
        generic_sm_log_cls_detached = [] # Only for generic modules
        global_log_surprise_gt_sum = 0.0
        valid_surprises = 0

        for module in all_modules_inc_taskhead:
             true_task = true_answer_c if module is self.task_head else None
             # calculate_surprise now returns grad-enabled loss component
             _ = module.calculate_surprise(ct_plus_1, true_task_output=true_task)

             # Store detached CLS log surprise (dummy for TaskHead)
             sm_log_cls_detached = module.last_log_surprise_sm_cls
             log_surprises_sm_cls_detached.append(sm_log_cls_detached)
             raw_surprises_sm_cls_detached.append(module.last_raw_surprise_sm_cls)

             # Accumulate global surprise based on CLS prediction (even dummy for TaskHead)
             if math.isfinite(sm_log_cls_detached.item()):
                 global_log_surprise_gt_sum += sm_log_cls_detached.item(); valid_surprises += 1
                 # Store generic module CLS surprise separately
                 if module is not self.task_head:
                     generic_sm_log_cls_detached.append(sm_log_cls_detached)

        # Calculate average global (CLS) log surprise
        global_log_surprise_gt = (global_log_surprise_gt_sum / valid_surprises) if valid_surprises > 0 else float('nan')
        # Calculate average generic module CLS log surprise
        sm_log_cls_avg_generic = torch.mean(torch.stack(generic_sm_log_cls_detached)) if generic_sm_log_cls_detached else torch.tensor(0.0, device=self.device)

        # --- 6. Estimate LE (Optional) ---
        lambda_max_estimate = None
        if estimate_le and torch.all(torch.isfinite(ct_prev_dense)):
            try:
                # Use current gamma_t for dynamics map
                dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
                lambda_max_estimate = estimate_lyapunov_exponent(dynamics_map_for_le, ct_prev_dense, device=self.device) # Pass device
                # Check if LE is valid
                if lambda_max_estimate is not None and not math.isfinite(lambda_max_estimate):
                     print(f"Warning: LE estimation returned non-finite value {lambda_max_estimate}")
                     lambda_max_estimate = None
            except Exception as le_error:
                 print(f"Error during LE estimation call: {le_error}")
                 traceback.print_exc()
                 lambda_max_estimate = None

        # --- 7. Update Meta-Model EMA Inputs ---
        # Update Gt EMA
        current_gt_log_val = global_log_surprise_gt
        if math.isfinite(current_gt_log_val):
            if self.gt_log_ema is None: self.gt_log_ema = current_gt_log_val
            else: self.gt_log_ema = (1 - self.ema_alpha) * current_gt_log_val + self.ema_alpha * self.gt_log_ema
        # Update Lambda EMA
        current_lambda_val = lambda_max_estimate
        # Use target if LE is invalid
        effective_lambda = current_lambda_val if current_lambda_val is not None else self.meta_model.stability_target
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

        for i, module in enumerate(all_modules_inc_taskhead):
            if module is self.task_head:
                # TaskHead learns only from its prediction loss (no policy reward signal needed)
                total_loss, policy_loss, pred_loss, entropy, grad_norm = module.learn(None, current_step_num)
            else:
                # Generic modules learn using their own CLS surprise as negative reward
                reward_signal = -log_surprises_sm_cls_detached[i] # Negative surprise = reward
                total_loss, policy_loss, pred_loss, entropy, grad_norm = module.learn(reward_signal, current_step_num)

            module_total_losses.append(total_loss)
            module_policy_losses.append(policy_loss)
            module_pred_losses.append(pred_loss)
            module_entropies.append(entropy)
            module_grad_norms.append(grad_norm)

        # --- 10. Update State ---
        self.ct = ct_plus_1

        # --- 11. Calculate Task Accuracy for Logging ---
        task_correct = 0
        task_prediction = self.task_head.get_task_prediction() # Get detached prediction
        if task_prediction is not None and true_answer_c is not None and \
           torch.isfinite(task_prediction) and torch.isfinite(true_answer_c):
            # Compare rounded prediction to true answer
            if abs(task_prediction.item() - true_answer_c.item()) < 0.5:
                task_correct = 1

        # --- 12. Return Metrics ---
        # Collect finite values for aggregation
        finite_sm_log_cls = [s.item() for s in log_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_sm_raw_cls = [s.item() for s in raw_surprises_sm_cls_detached if s is not None and math.isfinite(s.item())]
        finite_total_losses = [l for l in module_total_losses if l is not None and math.isfinite(l)]
        finite_policy_losses = [l for l in module_policy_losses if l is not None and math.isfinite(l)]
        finite_pred_losses = [l for l in module_pred_losses if l is not None and math.isfinite(l)]
        finite_entropies = [e for e in module_entropies if e is not None and math.isfinite(e)]
        finite_module_grads = [g for g in module_grad_norms if g is not None and math.isfinite(g)]

        # Get task-specific surprises (log and raw)
        task_sm_log = self.task_head.last_log_surprise_sm_task.item() if self.task_head.last_log_surprise_sm_task is not None else float('nan')
        task_sm_raw = self.task_head.last_raw_surprise_sm_task.item() if self.task_head.last_raw_surprise_sm_task is not None else float('nan')

        metrics = {
            "Gt_log": global_log_surprise_gt, # Already checked for finite
            "Gt_log_EMA": self.gt_log_ema,
            "Sm_log_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_log_std": np.std(finite_sm_log_cls) if len(finite_sm_log_cls) > 1 else 0.0,
            "Sm_log_cls_avg": np.mean(finite_sm_log_cls) if finite_sm_log_cls else float('nan'),
            "Sm_log_cls_avg_generic": sm_log_cls_avg_generic.item() if torch.is_tensor(sm_log_cls_avg_generic) else sm_log_cls_avg_generic,
            "Sm_raw_cls_avg": np.mean(finite_sm_raw_cls) if finite_sm_raw_cls else float('nan'),
            "TaskHead_Sm_log_task": task_sm_log if math.isfinite(task_sm_log) else float('nan'),
            "TaskHead_Sm_raw_task": task_sm_raw if math.isfinite(task_sm_raw) else float('nan'),
            "lambda_max_est": lambda_max_estimate, # Can be None
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
            "cls_density": self.ct._nnz() / self.cls_dim if self.ct._nnz() is not None and self.cls_dim > 0 else float('nan'),
            "TaskCorrect": task_correct
        }
        return metrics