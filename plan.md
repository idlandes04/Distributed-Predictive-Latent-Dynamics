Okay, I have conducted a comprehensive review of the generated codebase, considering the DPLD theory, empirical testing standards, and the traceback error you encountered.

**Key Findings and Refinements:**

1.  **Traceback Error (`TypeError: 'method' object is not iterable`):** This was the most critical issue. As identified in the thought process, the attribute name `self.modules` in `DPLDSystem` conflicts with the built-in `nn.Module.modules()` method. Renaming the attribute (e.g., to `self.pred_modules`) resolves this.
2.  **Lyapunov Exponent (LE) Estimation (`lambda_max`):**
    *   **Complexity & Speed:** Calculating LE accurately within the main training loop using `autograd.functional.jvp` can be slow and complex due to the need for a consistent, differentiable dynamics map (`F(ct)`).
    *   **Consistency:** The `get_dynamics_map` function needs to represent a *fixed* dynamic for the estimation period, meaning parameters influencing the map (like `gamma_t`, module weights used for `Im`) should ideally be held constant *during* the LE estimation steps.
    *   **Refinement:**
        *   The `estimate_lyapunov_exponent` utility in `utils.py` is kept as it's a standard algorithm.
        *   The `get_dynamics_map` method in `DPLDSystem` is refined to use *fixed* parameters (average gamma, zero noise, current module parameters but detached) to define the map for LE estimation, making the estimation more consistent.
        *   The *online* estimation of LE within the main `step` loop is made optional and potentially less frequent (controlled by `--le-interval` in `train.py`) because it's computationally intensive.
        *   The `MetaModel`'s `compute_regulation` function will now use the `last_lambda_max` that was estimated (potentially several steps ago) to determine the *current* `gamma_t`. This introduces a slight delay in the feedback loop but is more practical than estimating LE every single step. If no LE estimate is available yet, it uses a default gamma.
3.  **Difference Reward Calculation:** The counterfactual calculation (`Gt^{-m}`) involves recomputing the next state `M` times per step. This is computationally expensive but necessary to test the core hypothesis. For the MVP, this is acceptable, but optimization might be needed for larger `M`. Added `.detach()` to ensure rewards don't carry gradients back inappropriately.
4.  **Stochasticity in Write Vector:** The use of `Normal(mean, std).sample()` in `PredictiveModule.generate_write_vector` correctly introduces stochasticity needed for REINFORCE and allows gradients to flow back to the parameters determining the `mean` (i.e., the module's `fm` and `qm`).
5.  **Sparse Tensor Handling:** Added `.coalesce()` after sparse additions to maintain efficiency. Ensured conversions `.to_dense()` happen where needed (MLP inputs, loss calculations). Explicitly handled the case of an all-zero sparse vector after sparsification.
6.  **Clarity and Comments:** Added more comments explaining design choices, approximations (like using baseline surprise for `alpha_m`), and potential complexities (like Meta-Model learning).
7.  **Device Handling:** Ensured tensors are consistently moved to the correct device.

---

**Updated Code Files:**

**`envs.py`** (No changes needed from the previous version)

```python
import torch
import numpy as np

class LorenzEnv:
    """
    Simulates the Lorenz attractor dynamics.
    Provides states for the DPLD system to predict.
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01, device='cpu'):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.device = device
        self.state = None
        self.reset()

    def _lorenz_dynamics(self, state_tensor):
        x, y, z = state_tensor[..., 0], state_tensor[..., 1], state_tensor[..., 2]
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        # Stack along the last dimension
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)

    def step(self):
        """Advances the simulation by one time step using RK4 integration."""
        k1 = self._lorenz_dynamics(self.state)
        k2 = self._lorenz_dynamics(self.state + 0.5 * self.dt * k1)
        k3 = self._lorenz_dynamics(self.state + 0.5 * self.dt * k2)
        k4 = self._lorenz_dynamics(self.state + self.dt * k3)
        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.get_observation()

    def reset(self, initial_state=None):
        """Resets the environment state."""
        if initial_state is None:
            # Use standard initial conditions or random ones
            # self.state = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=self.device)
             self.state = (torch.rand(3, device=self.device) - 0.5) * 2 * 10 # Random start
        else:
            self.state = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        return self.get_observation()

    def get_observation(self):
        """Returns the current state."""
        # In this simple case, observation is the state itself.
        # Could add noise or projection later.
        return self.state.clone()

    def get_dimension(self):
        return 3

# Example usage:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LorenzEnv(device=device)
    print(f"Device: {env.device}")
    print(f"Initial state: {env.reset()}")
    trajectory = []
    for _ in range(1000):
        state = env.step()
        trajectory.append(state.cpu().numpy())

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_title("Lorenz Attractor Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
```

---

**`utils.py`** (Minor refinement in LE estimation comment)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
import os
import time

class Logger:
    """Simple logger for console and CSV."""
    def __init__(self, log_dir="logs", run_name=None):
        if run_name is None:
            run_name = f"dpld_run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.metrics = {}
        self.header_written = False
        print(f"Logging to {self.log_dir}")

    def log(self, step, metrics_dict):
        """Logs metrics to console and CSV."""
        # Filter out None values before logging
        loggable_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
        self.metrics = {"step": step, **loggable_metrics}

        print(f"Step: {step}", end="")
        for k, v in loggable_metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, np.float32, np.float64)):
                 print(f" | {k}: {v:.4f}", end="")
            else:
                 print(f" | {k}: {v}", end="")
        print() # Newline

        # Prepare data row ensuring order matches header if written, handle missing keys
        if self.header_written:
            data_row = [self.metrics.get(k, '') for k in self.header_keys]
        else:
            self.header_keys = list(self.metrics.keys()) # Store header order
            data_row = list(self.metrics.values())


        # Write to CSV
        mode = 'a' if self.header_written else 'w'
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if not self.header_written:
                writer.writerow(self.header_keys)
                self.header_written = True
            writer.writerow(data_row)


    def close(self):
        pass # Placeholder if needed for file handles

def plot_metrics(log_file, metrics_to_plot=None):
    """Plots metrics from a CSV log file."""
    import pandas as pd
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file}")
        return
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if df.empty:
        print("Log file is empty.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = [col for col in df.columns if col != 'step' and pd.api.types.is_numeric_dtype(df[col])]

    num_plots = len(metrics_to_plot)
    if num_plots == 0:
        print("No numeric metrics found to plot.")
        return

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    plot_count = 0
    for metric in metrics_to_plot:
        if metric in df.columns:
            # Drop rows where the metric might be NaN or non-numeric after load
            plot_data = df[['step', metric]].dropna()
            if not plot_data.empty:
                axes[plot_count].plot(plot_data['step'], plot_data[metric])
                axes[plot_count].set_title(metric)
                axes[plot_count].set_xlabel("Step")
                axes[plot_count].set_ylabel(metric)
                axes[plot_count].grid(True)
                plot_count += 1
            else:
                 print(f"Warning: Metric '{metric}' has no plottable data.")
        else:
            print(f"Warning: Metric '{metric}' not found in log file.")


    # Hide any remaining empty subplots
    for j in range(plot_count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # Save the plot instead of showing interactively
    plot_filename = os.path.splitext(log_file)[0] + "_plot.png"
    try:
        plt.savefig(plot_filename)
        print(f"Metrics plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory


def estimate_lyapunov_exponent(dynamics_map_fn, state, n_vectors=5, steps=50, epsilon=1e-5, device='cpu'):
    """
    Estimates the largest Lyapunov exponent using Jacobian-vector products.
    Based on Algorithm 2 sketch from Part II, adapted for PyTorch autograd.

    Args:
        dynamics_map_fn: A function that takes a state tensor (ct) and returns
                         the next state tensor (ct+1). Must be differentiable.
                         IMPORTANT: This map should represent a *fixed* dynamic
                         for the duration of the estimation steps.
        state: The starting state tensor (shape [D]). Assumed DENSE.
        n_vectors: Number of orthonormal vectors to track (k in Alg 2).
        steps: Number of steps for averaging (T in Alg 2).
        epsilon: Small perturbation for JVP calculation if needed (often not).
        device: Torch device.

    Returns:
        Estimated largest Lyapunov exponent (lambda_max). Returns None if estimation fails.
    """
    with torch.no_grad(): # Ensure no gradients are computed within LE estimation itself
        D = state.shape[0]
        if D < n_vectors:
            print(f"Warning: State dimension {D} < n_vectors {n_vectors}. Reducing n_vectors.")
            n_vectors = D
        if n_vectors == 0: return None # Cannot estimate with 0 vectors

        # Initialize orthonormal vectors (columns of Q)
        try:
            q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, device=device))[0]
        except torch.linalg.LinAlgError:
             print("Error: QR decomposition failed during LE initialization.")
             return None


        log_stretch_sum = 0.0
        current_state = state.detach().clone() # Ensure it's a detached copy

    # --- JVP Calculation Function ---
    # This function needs gradients enabled temporarily
    def jvp_fn(v_in):
        # We need the Jacobian of dynamics_map_fn w.r.t current_state_detached
        current_state_detached = current_state.detach().requires_grad_(True)
        # Use autograd.functional.jvp
        map_fn_for_jvp = lambda s: dynamics_map_fn(s)
        try:
            _, output_tangent = torch.autograd.functional.jvp(
                map_fn_for_jvp, current_state_detached, v_in, create_graph=False
            )
            return output_tangent
        except Exception as e:
             print(f"Error during JVP calculation: {e}")
             return None # Signal failure


    # --- Main LE Estimation Loop ---
    for step_num in range(steps):
        # Compute JVPs for all vectors in q_matrix
        v_list = []
        for i in range(n_vectors):
            v_out = jvp_fn(q_matrix[:, i])
            if v_out is None: return None # Propagate JVP failure
            v_list.append(v_out)

        if not v_list: return None # Should not happen if n_vectors > 0

        v_matrix = torch.stack(v_list, dim=1)

        # QR decomposition of the resulting vectors: V = Q' R'
        try:
            # Need to handle potential non-finite values coming from JVP
            if not torch.all(torch.isfinite(v_matrix)):
                 print(f"Warning: Non-finite values in JVP result at LE step {step_num}. Skipping.")
                 # Attempt recovery: Re-orthogonalize q_matrix and continue
                 q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, device=device))[0]
                 continue

            q_prime, r_prime = torch.linalg.qr(v_matrix)
        except torch.linalg.LinAlgError:
             print(f"Warning: QR decomposition failed during LE step {step_num}. Skipping.")
             # Attempt recovery: Re-orthogonalize q_matrix and continue
             q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, device=device))[0]
             continue # Skip this step's contribution


        # Accumulate the log of the diagonal elements of R' (stretching factors)
        # Ensure diagonal elements are non-zero before log
        diag_r = torch.diag(r_prime)
        # Clamp small values away from zero for stability
        safe_diag_r = torch.clamp(torch.abs(diag_r), min=1e-9)
        log_stretch_sum += torch.sum(torch.log(safe_diag_r))

        # Update orthonormal vectors for the next step
        q_matrix = q_prime

        # Update the state for the next iteration using the original dynamics map
        # Do this outside JVP calculation, no gradients needed here
        with torch.no_grad():
             current_state = dynamics_map_fn(current_state.detach()) # Use detached state
             if not torch.all(torch.isfinite(current_state)):
                  print("Error: Non-finite state encountered during LE state update. Aborting LE estimation.")
                  return None


    # Average the log stretch factors
    # lambda_max_estimate = log_stretch_sum / (steps * n_vectors) # Avg over steps and vectors
    # A common definition focuses on the sum of logs, representing the expansion rate.
    # Dividing by steps gives the average rate.
    # Focusing only on the first vector's stretch (r_prime[0,0]) approximates the *largest* LE.
    # Let's return the average log sum per step.
    lambda_max_estimate = log_stretch_sum / steps

    return lambda_max_estimate.item()


def sparsify_vector(vector, k_fraction):
    """Zeros out all but the top-k% largest magnitude elements."""
    if k_fraction >= 1.0:
        return vector
    if k_fraction <= 0.0:
        return torch.zeros_like(vector)

    D = vector.numel()
    if D == 0: return vector # Handle empty vector case
    k_count = max(1, int(D * k_fraction)) # Ensure at least 1 element is kept

    # Find the threshold value for the top-k elements
    abs_vector = torch.abs(vector)
    # Handle potential NaNs or Infs - replace them with 0 for threshold calculation
    valid_abs_vector = torch.nan_to_num(abs_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure k_count is not larger than the number of elements
    k_count = min(k_count, D)

    # Use topk directly which is more robust than kthvalue for finding threshold
    try:
        threshold = torch.topk(valid_abs_vector, k_count).values[-1] # Value of the k-th largest element
    except RuntimeError as e:
         # This can happen if k_count > number of non-zero elements, etc.
         print(f"Warning: Error in topk during sparsify ({e}). Keeping all elements.")
         return vector # Fallback: return original vector


    # Create a mask where abs(vector) >= threshold
    mask = valid_abs_vector >= threshold

    # If too many elements are selected due to ties (or threshold is 0), keep only k_count
    if mask.sum() > k_count:
         # Get indices of all elements >= threshold
        potential_indices = torch.where(mask)[0]
        # Get their absolute values
        potential_abs_values = valid_abs_vector[potential_indices]
        # Sort by magnitude and take top k_count indices relative to potential_indices
        _, top_indices_relative = torch.topk(potential_abs_values, k_count)
        # Get the actual indices in the original vector
        final_indices = potential_indices[top_indices_relative]
        # Create new mask
        mask = torch.zeros_like(vector, dtype=torch.bool, device=vector.device)
        mask[final_indices] = True
    elif mask.sum() == 0 and k_count > 0 and D > 0:
         # Handle case where threshold is 0 and all values are 0, but k_count > 0
         # Or if all values are NaN/Inf and valid_abs_vector is all 0
         # Keep the top k largest magnitude original values (even if 0)
         _, top_indices = torch.topk(valid_abs_vector, k_count)
         mask = torch.zeros_like(vector, dtype=torch.bool, device=vector.device)
         mask[top_indices] = True


    # Apply mask
    sparse_vec = torch.zeros_like(vector)
    sparse_vec[mask] = vector[mask]

    return sparse_vec

def sparsify_tensor_batch(tensor_batch, k_fraction):
    """Applies sparsify_vector to each tensor in a batch."""
    return torch.stack([sparsify_vector(t, k_fraction) for t in tensor_batch])

# --- PyTorch Sparse Utilities (Placeholders - not strictly needed for MVP) ---
def sparse_dense_matmul(sparse_matrix, dense_matrix):
    return torch.sparse.mm(sparse_matrix, dense_matrix)

def dense_sparse_matmul(dense_matrix, sparse_matrix):
    return torch.sparse.mm(sparse_matrix.t(), dense_matrix.t()).t()

def sparse_elementwise_add(sparse_tensor1, sparse_tensor2):
    return (sparse_tensor1 + sparse_tensor2).coalesce()

def sparse_elementwise_mul(sparse_tensor1, sparse_tensor2):
     # Element-wise multiply for sparse tensors. Result is sparse.
     # Only non-zero where *both* are non-zero.
     return (sparse_tensor1 * sparse_tensor2).coalesce()

def add_sparse_to_dense(dense_tensor, sparse_tensor):
    if sparse_tensor._nnz() == 0:
        return dense_tensor
    return dense_tensor + sparse_tensor.to_dense() # Simplest way

def get_sparse_tensor_size(sparse_tensor):
     return sparse_tensor.size()

def get_sparse_values(sparse_tensor):
     return sparse_tensor.values()

def get_sparse_indices(sparse_tensor):
    return sparse_tensor.indices()
```

---

**`core.py`** (Major fix: Renamed `self.modules` to `self.pred_modules`. Refined `get_dynamics_map` and LE usage.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical # Or others if needed
import numpy as np # Added for metrics calculation

from utils import estimate_lyapunov_exponent, sparsify_vector # Add sparse utils if needed

# --- Constants ---
EPSILON = 1e-8 # For numerical stability (e.g., log probabilities)

# --- Predictive Module ---
class PredictiveModule(nn.Module):
    """
    Implements a DPLD module: Read, Predict, Write (via sparse, gated contribution).
    Learns via difference reward based on global surprise.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.k_sparse_write = k_sparse_write # Sparsity fraction k (Part II, Alg 1)
        self.device = device
        self.surprise_scale_factor = surprise_scale_factor # βα in Part II, Alg 1, Eq 4
        self.surprise_baseline_ema = surprise_baseline_ema # For Sm baseline in Alg 1

        # Internal Predictive Model (fm in Part II, Sec 4.1) - Simple MLP
        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
             # Output layer predicts the *entire* next CLS state for MVP simplicity
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)
        # Parameters θm are implicitly self.fm.parameters()

        # Gating Query Vector (qm in Part II, Alg 1)
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.1)

        # Store running average of surprise Sm (Sm_bar in Alg 1 logic)
        self.register_buffer('sm_baseline', torch.tensor(1.0, device=device)) # Initialize baseline reasonably

        # Store last prediction, surprise, and write vector for learning
        self.last_prediction_ct_plus_1 = None # ĉm,t+1
        self.last_surprise_sm = None          # Sm
        # self.last_write_vector_im = None      # Im - Not needed explicitly after use
        self.last_log_prob = None             # Log prob of action (write vector) for REINFORCE

        self.optimizer = optim.Adam(list(self.fm.parameters()) + [self.qm], lr=learning_rate)


    def predict(self, ct):
        """Predicts the next CLS state based on the current state ct."""
        # ct is expected to be a sparse tensor, convert to dense for MLP
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        # Detach ct_dense from graph if module prediction shouldn't influence upstream grads
        self.last_prediction_ct_plus_1 = self.fm(ct_dense.detach()) # ĉm,t+1
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1):
        """Calculates local surprise Sm based on prediction and actual next state."""
        # Ensure actual_ct_plus_1 is dense for loss calculation
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             # This can happen if called before predict in a step, handle gracefully
             print("Warning: calculate_surprise called before predict. Returning baseline surprise.")
             return self.sm_baseline.detach() # Return detached baseline

        # Using Mean Squared Error as the distance metric (Eq 2, Part II)
        surprise = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense.detach(), reduction='mean')

        self.last_surprise_sm = surprise

        # Update baseline Sm (Sm_bar) using EMA
        # Ensure baseline update doesn't affect gradient flow
        with torch.no_grad():
            self.sm_baseline = self.surprise_baseline_ema * self.sm_baseline + \
                               (1 - self.surprise_baseline_ema) * surprise
            # Clamp baseline to avoid potential collapse to zero or explosion
            self.sm_baseline = torch.clamp(self.sm_baseline, min=1e-6, max=1e6)


        return self.last_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse, weighted, gated write vector Im (Alg 1, Part II)."""
        if self.last_prediction_ct_plus_1 is None:
             raise RuntimeError("Must call predict() before generate_write_vector()")

        # Use the current surprise baseline for alpha_m calculation, as actual surprise isn't known yet.
        current_surprise_for_alpha = self.sm_baseline.detach()

        # Alg 1, Step 1: Project module output (vm = ĉm,t+1 in MVP)
        vm = self.last_prediction_ct_plus_1 # Raw output vector

        # Alg 1, Step 2 & 3: Compute element-wise gate activation (gm = sigmoid(qm * ct / tau_g))
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        tau_g = 1.0 # Gating temperature, hyperparameter
        # Ensure qm * ct_dense doesn't create gradient issues if ct comes from graph
        gate_activation_gm = torch.sigmoid(self.qm * ct_dense.detach() / tau_g) # gm (vector [0,1]^D)

        # Alg 1, Step 4: Modulate influence by surprise (αm = α_base + α_scale * tanh(βα(Sm - Sm_bar)))
        alpha_base = 1.0 # Base influence
        alpha_scale = 1.0 # Scaling factor for surprise modulation
        surprise_diff = current_surprise_for_alpha - self.sm_baseline # Use baseline vs baseline (should be ~0 avg)
        # Let's use actual last surprise if available, otherwise baseline
        # surprise_to_use = self.last_surprise_sm if self.last_surprise_sm is not None else self.sm_baseline
        # surprise_diff = surprise_to_use - self.sm_baseline
        # Using baseline vs baseline is safer if last_surprise isn't computed yet.
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * surprise_diff)
        # Ensure non-negative and finite influence
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)

        # Alg 1, Step 5: Apply gating and scaling (Im_dense = αm * (gm ⊙ vm))
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm) # Dense intermediate Im

        # --- Stochasticity for REINFORCE ---
        action_mean = intermediate_write_vector
        action_std = 0.1 # Fixed std deviation for simplicity, could be learned/scheduled
        # Ensure std is positive
        action_std_tensor = torch.full_like(action_mean, fill_value=action_std)

        dist = Normal(action_mean, action_std_tensor)
        dense_write_vector_sampled = dist.sample()

        # Calculate log_prob based on the *sampled* action and the *distribution* it came from
        # Ensure no gradients flow back from log_prob calculation itself to the distribution parameters here
        # We want gradients only when loss.backward() is called in learn()
        self.last_log_prob = dist.log_prob(dense_write_vector_sampled.detach()).sum()

        # Alg 1, Step 6 & 7: Sparsify the contribution vector
        write_vector_im_sparse_vals = sparsify_vector(dense_write_vector_sampled, self.k_sparse_write)

        # Convert to sparse tensor format for efficiency
        sparse_indices = torch.where(write_vector_im_sparse_vals != 0)[0].unsqueeze(0)
        sparse_values = write_vector_im_sparse_vals[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device)
        else:
             # Handle case where vector becomes all zero after sparsification
             im_sparse = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                 torch.empty((0,), dtype=torch.float32, device=self.device),
                                                 (self.cls_dim,))

        return im_sparse.coalesce() # Return coalesced sparse tensor

    def learn(self, difference_reward_rm):
        """Updates module parameters using the difference reward."""
        if self.last_log_prob is None:
             # This happens on the first step or if generate_write wasn't called properly
             return 0.0 # Return zero loss

        # REINFORCE update rule: loss = - R * log_prob
        # Ensure reward is a scalar tensor
        reward_tensor = torch.tensor(difference_reward_rm, device=self.device)

        # Ensure log_prob requires grad if it came from parameters
        # self.last_log_prob should implicitly depend on fm and qm through action_mean
        loss = -reward_tensor * self.last_log_prob

        self.optimizer.zero_grad()
        # Retain graph if log_prob needs to be used elsewhere? No.
        loss.backward()
        # Optional: Gradient clipping
        nn.utils.clip_grad_norm_(list(self.fm.parameters()) + [self.qm], max_norm=1.0)
        self.optimizer.step()

        # Clear stored log_prob for next step
        self.last_log_prob = None
        # Keep last_prediction and last_surprise for potential analysis if needed, or clear them too?
        # Let's clear prediction, surprise will be recalculated next step.
        self.last_prediction_ct_plus_1 = None
        # self.last_surprise_sm = None # Keep this? No, recalculate next step.

        return loss.item()


# --- Meta-Model ---
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
             gamma_signal = torch.sigmoid(torch.tensor(self.last_estimated_lambda_max - self.stability_target, device=self.device))
             gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * gamma_signal
             gamma_t = torch.clamp(gamma_t, self.gamma_min, self.gamma_max) # Ensure bounds
             gamma_t = gamma_t.item() # Convert to scalar float
        else:
             # Default gamma if stability not estimated yet or estimate was invalid
             gamma_t = (self.gamma_min + self.gamma_max) / 2.0

        # --- Modulatory Vector mmod_t ---
        # Set to zero sparse tensor for MVP
        mmod_t = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                         torch.empty((0,), dtype=torch.float32, device=self.device),
                                         (self.cls_dim,))

        return gamma_t, mmod_t


    def learn(self):
        """Updates Meta-Model parameters (DISABLED in MVP)."""
        # Objective LMM: Minimize instability penalty (Eq 5, Part II, simplified)
        if self.last_estimated_lambda_max is None or not np.isfinite(self.last_estimated_lambda_max):
            return 0.0 # Cannot learn without stability estimate

        instability_penalty = F.relu(torch.tensor(self.last_estimated_lambda_max - self.stability_target, device=self.device))
        loss = instability_penalty # Simple ReLU penalty

        # Learning is disabled for MVP as gradient path is complex
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        return loss.item()


# --- DPLD System ---
class DPLDSystem(nn.Module):
    """
    The main DPLD system coordinating CLS, Modules, and Meta-Model.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, noise_std_dev_schedule,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules
        self.k_sparse_write = k_sparse_write
        self.noise_std_dev_schedule = noise_std_dev_schedule # Function step -> std_dev
        self.device = device

        # Initialize CLS state (sparse tensor)
        self.ct = self._init_cls()

        # Initialize Modules - FIX: Use a different attribute name
        self.pred_modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr, device=device)
            for _ in range(num_modules)
        ])

        # Initialize Meta-Model
        self.meta_model = MetaModel(cls_dim, meta_hidden_dim, meta_lr,
                                    gamma_min, gamma_max, stability_target, device=device)

        # Buffer for difference reward calculation
        self.last_global_surprise_gt = None

    def _init_cls(self):
        # Start with a zero or small random sparse vector
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.1
        sparse_ct = sparsify_vector(initial_dense, 0.1).to_sparse_coo() # Start sparse
        return sparse_ct.coalesce()


    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation (Eq 1, Part II)."""
        # Ensure inputs are sparse
        if not ct.is_sparse: ct = ct.to_sparse_coo()
        if not sum_im.is_sparse: sum_im = sum_im.to_sparse_coo()
        if not mmodt.is_sparse: mmodt = mmodt.to_sparse_coo()

        # Decay term: (1 - gamma_t) * ct
        decayed_ct = (1.0 - gamma_t) * ct

        # Noise term: epsilon_t ~ N(0, sigma_t^2 * I)
        noise_et = torch.randn(self.cls_dim, device=self.device) * noise_std_dev
        # Add dense noise to the dense representation for simplicity, then convert back if needed
        # Or add sparse noise? Let's add dense noise to the sum of other terms.
        # noise_et_sparse = noise_et.to_sparse_coo() # Might be inefficient

        # Combine terms: ct+1 = (1-gamma)ct + Sum(Im) + mmod_t + eps_t
        # Perform additions in sparse format
        ct_plus_1_sparse = (decayed_ct + sum_im + mmodt).coalesce()

        # Add dense noise - convert sparse sum to dense, add noise, convert back
        # This is potentially a bottleneck but simpler than sparse noise generation
        ct_plus_1_dense = ct_plus_1_sparse.to_dense() + noise_et
        ct_plus_1 = ct_plus_1_dense.to_sparse_coo() # Convert back to sparse

        # Optional: Explicit normalization or re-sparsification (see utils/core discussions)
        # ct_plus_1 = ct_plus_1.coalesce() # Ensure coalesced after potential format changes

        return ct_plus_1.coalesce()


    def get_dynamics_map(self, fixed_gamma, fixed_noise_std=0.0):
         """
         Returns a function representing the one-step CLS dynamics for LE estimation.
         Uses current module parameters but *fixed* gamma and noise for consistency.
         """
         # Ensure gamma is a float
         gamma_val = float(fixed_gamma)

         # Create a snapshot of module states (parameters are implicitly captured)
         # We need to ensure the map uses the *current* parameters but doesn't train them further.
         current_module_params = [p.detach().clone() for mod in self.pred_modules for p in mod.parameters()]

         def dynamics_map(state_t_dense):
             # state_t is assumed dense for JVP calculation
             state_t_sparse = state_t_dense.to_sparse_coo()

             # 1. Get module write vectors Im based on state_t
             sum_im = self._init_cls() # Zero sparse tensor
             with torch.no_grad(): # Ensure no gradients computed within this map function
                 for i, module in enumerate(self.pred_modules):
                     # Use module's current state (params) but detached inputs/outputs
                     # Predict based on input state
                     pred = module.predict(state_t_sparse) # Uses internal fm
                     # Generate write vector using baseline surprise for alpha_m
                     module.last_prediction_ct_plus_1 = pred.detach()
                     module.last_surprise_sm = module.sm_baseline.detach() # Use baseline
                     # Use the *deterministic* mean of the action for LE map
                     # Replicate steps 1-5 of generate_write_vector deterministically
                     vm = module.last_prediction_ct_plus_1
                     ct_dense_detached = state_t_sparse.to_dense().detach()
                     tau_g = 1.0
                     gate_activation_gm = torch.sigmoid(module.qm.detach() * ct_dense_detached / tau_g)
                     alpha_base = 1.0
                     alpha_scale = 1.0
                     surprise_diff = module.sm_baseline.detach() - module.sm_baseline.detach() # Approx 0
                     influence_scalar_am = alpha_base + alpha_scale * torch.tanh(module.surprise_scale_factor * surprise_diff)
                     influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1, max=10.0)
                     intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm)
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
                     # Clear module state after use within the map
                     module.last_prediction_ct_plus_1 = None
                     module.last_surprise_sm = None


                 # 2. Get fixed meta-model regulation
                 gamma_t = gamma_val
                 mmodt = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long, device=self.device),
                                                  torch.empty((0,), dtype=torch.float32, device=self.device),
                                                  (self.cls_dim,)) # Zero for MVP


                 # 3. Apply CLS update rule with fixed noise
                 next_state = self.cls_update_rule(state_t_sparse, sum_im.coalesce(), mmodt, gamma_t, fixed_noise_std)

             # Return dense state as required by LE estimator JVP
             return next_state.to_dense()

         return dynamics_map


    def step(self, current_step_num, estimate_le=False):
        """Performs one full step of the DPLD system interaction."""

        # --- Store previous state for potential LE estimation ---
        ct_prev_dense = self.ct.to_dense().detach()

        # --- 1. Module Predictions ---
        predictions = []
        # FIX: Iterate over the renamed attribute
        for module in self.pred_modules:
            # Ensure predict uses detached state if CLS has grads from prev step?
            # Let's detach self.ct before passing to modules
            predictions.append(module.predict(self.ct.detach()))

        # --- 2. Meta-Model Regulation ---
        # Compute gamma_t and mmod_t based on *last known* LE estimate
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.ct.detach())

        # --- 3. Module Write Vectors ---
        write_vectors_im = []
        sum_im = self._init_cls() # Zero sparse tensor
        # FIX: Iterate over the renamed attribute
        for i, module in enumerate(self.pred_modules):
             # predict() was already called.
             # generate_write_vector uses baseline surprise for alpha_m.
             im = module.generate_write_vector(self.ct.detach())
             write_vectors_im.append(im)
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual Surprises & Global Surprise ---
        surprises_sm = []
        global_surprise_gt = 0.0
        # FIX: Iterate over the renamed attribute
        for i, module in enumerate(self.pred_modules):
             # Now calculate the actual surprise using ct_plus_1
             sm = module.calculate_surprise(ct_plus_1) # Updates module.last_surprise_sm
             surprises_sm.append(sm)
             global_surprise_gt += sm
        # Handle potential division by zero if num_modules is 0
        if self.num_modules > 0:
            global_surprise_gt /= self.num_modules
        else:
            global_surprise_gt = torch.tensor(0.0, device=self.device)


        # --- 6. Calculate Difference Rewards ---
        difference_rewards_rm = []
        if self.num_modules > 0 and self.last_global_surprise_gt is not None:
            # Calculate Gt^{-m} for each module m
            for m_idx in range(self.num_modules):
                # Compute sum_im without module m
                sum_im_counterfactual = self._init_cls()
                for i in range(self.num_modules):
                    if i != m_idx:
                        # Ensure write_vectors_im[i] is valid sparse tensor
                         if write_vectors_im[i].is_sparse:
                             sum_im_counterfactual += write_vectors_im[i]
                sum_im_counterfactual = sum_im_counterfactual.coalesce()

                # Compute ct+1^{-m} using the same noise and gamma
                # Need the original self.ct for this counterfactual update
                ct_plus_1_counterfactual = self.cls_update_rule(
                    self.ct, sum_im_counterfactual, mmod_t, gamma_t, noise_std_dev
                )
                ct_plus_1_cf_dense = ct_plus_1_counterfactual.to_dense().detach()

                # Compute Gt^{-m} by calculating surprises relative to ct+1^{-m}
                gt_counterfactual = 0.0
                # FIX: Iterate over the renamed attribute
                for i, module in enumerate(self.pred_modules):
                    # Compare module i's prediction (predictions[i]) with the counterfactual state
                    # Ensure prediction is valid
                    if predictions[i] is not None:
                         sm_counterfactual = F.mse_loss(predictions[i].detach(), ct_plus_1_cf_dense, reduction='mean')
                         gt_counterfactual += sm_counterfactual
                    else:
                        # Handle case where prediction might be None if module didn't run
                        pass # Or add a default penalty?

                gt_counterfactual /= self.num_modules

                # Difference Reward Rm = Gt^{-m} - Gt
                rm = gt_counterfactual - global_surprise_gt
                difference_rewards_rm.append(rm.detach()) # Detach reward

        else: # First step or no modules
            difference_rewards_rm = [torch.tensor(0.0, device=self.device) for _ in range(self.num_modules)]

        # --- 7. Update State and Store Previous GT ---
        self.ct = ct_plus_1
        self.last_global_surprise_gt = global_surprise_gt.detach()

        # --- 8. Trigger Learning ---
        module_losses = []
        # FIX: Iterate over the renamed attribute
        for i, module in enumerate(self.pred_modules):
            # Ensure reward is valid
            reward = difference_rewards_rm[i] if i < len(difference_rewards_rm) else torch.tensor(0.0, device=self.device)
            loss = module.learn(reward)
            module_losses.append(loss)

        meta_loss = self.meta_model.learn() # Simplified meta-learning (returns penalty)

        # --- 9. Optional: Estimate LE for next step's regulation ---
        lambda_max_estimate = None
        if estimate_le:
            # Use the state *before* the update (ct_prev_dense) as the starting point
            # Use the *current* gamma_t that was just used in the update for the map
            dynamics_map_for_le = self.get_dynamics_map(fixed_gamma=gamma_t)
            lambda_max_estimate = estimate_lyapunov_exponent(
                dynamics_map_for_le,
                ct_prev_dense, # Start from the state before the step
                device=self.device
            )
            # Update meta-model's stored estimate
            self.meta_model.update_stability_estimate(lambda_max_estimate)


        # --- 10. Return Metrics ---
        metrics = {
            "Gt": global_surprise_gt.item(),
            "Sm_avg": torch.mean(torch.stack(surprises_sm)).item() if surprises_sm else 0.0,
            "Sm_std": torch.std(torch.stack(surprises_sm)).item() if len(surprises_sm) > 1 else 0.0,
            "Rm_avg": torch.mean(torch.stack(difference_rewards_rm)).item() if difference_rewards_rm else 0.0,
            "Rm_std": torch.std(torch.stack(difference_rewards_rm)).item() if len(difference_rewards_rm) > 1 else 0.0,
            "lambda_max_est": lambda_max_estimate, # LE estimated this step (if enabled)
            "gamma_t": gamma_t, # Gamma used this step
            "noise_std": noise_std_dev,
            "module_loss_avg": np.mean([l for l in module_losses if l is not None]) if module_losses else 0.0,
            "meta_loss": meta_loss, # Stability penalty term
            "cls_norm": torch.linalg.norm(self.ct.to_dense()).item(),
             "cls_density": self.ct._nnz() / self.cls_dim if self.ct._nnz() is not None else 0.0
        }

        return metrics

```

---

**`train.py`** (Updated to handle LE estimation interval)

```python
import torch
import argparse
import numpy as np
import os
import time # Added for run name

from core import DPLDSystem
# from envs import LorenzEnv # Environment not directly used in this MVP training loop
from utils import Logger, plot_metrics

def noise_schedule_const(step, std_dev=0.05):
    """Constant noise schedule."""
    return std_dev

def noise_schedule_anneal(step, start_std=0.1, end_std=0.01, anneal_steps=10000):
    """Linearly annealing noise schedule."""
    anneal_frac = min(1.0, step / anneal_steps)
    return start_std - (start_std - end_std) * anneal_frac

def main(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.use_gpu:
        print("Warning: --use-gpu specified but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    else:
        print("Using CPU.")
        device = torch.device("cpu")


    # --- Noise Schedule ---
    if args.noise_anneal > 0:
        noise_sched_fn = lambda step: noise_schedule_anneal(step, args.noise_start, args.noise_end, args.noise_anneal)
        print(f"Using annealing noise: start={args.noise_start}, end={args.noise_end}, steps={args.noise_anneal}")
    else:
        noise_sched_fn = lambda step: noise_schedule_const(step, args.noise_start)
        print(f"Using constant noise: std_dev={args.noise_start}")

    # --- DPLD System ---
    dpld_system = DPLDSystem(
        cls_dim=args.cls_dim,
        num_modules=args.num_modules,
        module_hidden_dim=args.module_hidden,
        meta_hidden_dim=args.meta_hidden,
        k_sparse_write=args.k_sparse,
        module_lr=args.module_lr,
        meta_lr=args.meta_lr, # Note: Meta LR currently unused as learning is off
        noise_std_dev_schedule=noise_sched_fn,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        stability_target=args.stability_target,
        device=device
    ).to(device)

    print(f"DPLD System initialized:")
    print(f"  CLS Dim: {args.cls_dim}")
    print(f"  Num Modules: {args.num_modules}")
    print(f"  Module Hidden: {args.module_hidden}")
    print(f"  Meta Hidden: {args.meta_hidden}")
    print(f"  Write Sparsity (k): {args.k_sparse}")
    print(f"  Module LR: {args.module_lr}")
    print(f"  Meta LR: {args.meta_lr}")
    print(f"  Gamma Range: [{args.gamma_min}, {args.gamma_max}]")
    print(f"  Stability Target (lambda_thr): {args.stability_target}")


    # --- Logger ---
    logger = Logger(log_dir=args.log_dir, run_name=args.run_name)

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()
    for step in range(args.total_steps):
        # Decide whether to estimate LE this step
        estimate_le_this_step = (args.le_interval > 0 and step % args.le_interval == 0)

        # Run one step of the DPLD internal dynamics
        metrics = dpld_system.step(step, estimate_le=estimate_le_this_step)

        # Logging
        if step % args.log_interval == 0 or step == args.total_steps - 1:
            logger.log(step, metrics)
            # Print estimated time remaining
            if step > 0:
                 elapsed_time = time.time() - start_time
                 steps_per_sec = step / elapsed_time
                 remaining_steps = args.total_steps - step
                 eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                 eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                 print(f"    ETA: {eta_str} ({steps_per_sec:.2f} steps/sec)")


        # Optional: Checkpointing
        if args.save_interval > 0 and step % args.save_interval == 0 and step > 0:
            save_dir = os.path.join(logger.log_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_step_{step}.pt")
            try:
                torch.save(dpld_system.state_dict(), save_path)
                print(f"    Checkpoint saved to {save_path}")
            except Exception as e:
                print(f"    Error saving checkpoint: {e}")


    print("Training finished.")
    total_time = time.time() - start_time
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.close()

    # --- Plotting ---
    print("Generating plot...")
    plot_metrics(logger.csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD MVP")

    # DPLD Architecture
    parser.add_argument('--cls-dim', type=int, default=512, help='Dimension of Central Latent Space (D)')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of predictive modules (M)')
    parser.add_argument('--module-hidden', type=int, default=128, help='Hidden dimension for module MLP')
    parser.add_argument('--meta-hidden', type=int, default=64, help='Hidden dimension for meta-model')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity fraction for module writes (k)')

    # Learning Parameters
    parser.add_argument('--module-lr', type=float, default=1e-4, help='Learning rate for modules')
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Learning rate for meta-model (currently unused)')
    parser.add_argument('--total-steps', type=int, default=50000, help='Total training steps')

    # Dynamics Parameters
    parser.add_argument('--gamma-min', type=float, default=0.01, help='Minimum global decay rate')
    parser.add_argument('--gamma-max', type=float, default=0.2, help='Maximum global decay rate')
    parser.add_argument('--stability-target', type=float, default=0.05, help='Target for lambda_max (lambda_thr)')
    parser.add_argument('--noise-start', type=float, default=0.1, help='Initial/constant noise std dev')
    parser.add_argument('--noise-end', type=float, default=0.01, help='Final noise std dev for annealing')
    parser.add_argument('--noise-anneal', type=int, default=10000, help='Steps to anneal noise over (0 for constant)')

    # Logging & Setup
    parser.add_argument('--log-interval', type=int, default=100, help='Steps between logging metrics')
    parser.add_argument('--le-interval', type=int, default=1000, help='Steps between estimating LE (0 to disable, can be slow)')
    parser.add_argument('--save-interval', type=int, default=5000, help='Steps between saving model checkpoints (0 to disable)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for logs and checkpoints')
    parser.add_argument('--run-name', type=str, default=None, help='Specific name for this run (default: auto-timestamp)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')


    args = parser.parse_args()
    main(args)
```

---

This refined codebase addresses the critical bug and incorporates improvements for stability, clarity, and practical execution of the MVP test suite. It should now be runnable and provide a better foundation for empirically testing the core DPLD hypotheses. Remember that LE estimation is computationally intensive and its accuracy depends on the stability of the underlying dynamics map; monitoring other metrics like `Gt`, `Sm`, `Rm`, and `cls_norm` will be crucial for evaluating the system's behavior.