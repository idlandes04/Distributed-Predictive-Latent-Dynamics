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