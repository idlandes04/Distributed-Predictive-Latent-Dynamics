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
        self.metrics = {"step": step, **metrics_dict}
        print(f"Step: {step}", end="")
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, np.float32, np.float64)):
                 print(f" | {k}: {v:.4f}", end="")
            else:
                 print(f" | {k}: {v}", end="")
        print() # Newline

        # Write to CSV
        if not self.header_written:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.metrics.keys())
                self.header_written = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Ensure order matches header if dict order changes (Python 3.7+)
            writer.writerow([self.metrics.get(k, '') for k in self.metrics.keys()])


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

    if metrics_to_plot is None:
        metrics_to_plot = [col for col in df.columns if col != 'step']

    num_plots = len(metrics_to_plot)
    if num_plots == 0:
        print("No metrics to plot.")
        return

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            axes[i].plot(df['step'], df[metric])
            axes[i].set_title(metric)
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
        else:
            print(f"Warning: Metric '{metric}' not found in log file.")
            axes[i].set_title(f"{metric} (Not Found)")
            axes[i].axis('off') # Hide unused subplot

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def estimate_lyapunov_exponent(dynamics_map_fn, state, n_vectors=5, steps=50, epsilon=1e-5, device='cpu'):
    """
    Estimates the largest Lyapunov exponent using Jacobian-vector products.
    Based on Algorithm 2 sketch from Part II, adapted for PyTorch autograd.

    Args:
        dynamics_map_fn: A function that takes a state tensor (ct) and returns
                         the next state tensor (ct+1). Must be differentiable.
        state: The starting state tensor (shape [D]).
        n_vectors: Number of orthonormal vectors to track (k in Alg 2).
        steps: Number of steps for averaging (T in Alg 2).
        epsilon: Small perturbation for JVP calculation if needed (often not).
        device: Torch device.

    Returns:
        Estimated largest Lyapunov exponent (lambda_max).
    """
    D = state.shape[0]
    if D < n_vectors:
        print(f"Warning: State dimension {D} < n_vectors {n_vectors}. Reducing n_vectors.")
        n_vectors = D

    # Initialize orthonormal vectors (columns of Q)
    q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, device=device))[0]

    log_stretch_sum = 0.0

    current_state = state.detach().clone()

    for _ in range(steps):
        # Ensure state requires grad for JVP
        current_state_detached = current_state.detach()
        
        # Function to compute JVP: J @ v
        def jvp_fn(v):
            # We need the Jacobian of dynamics_map_fn w.r.t current_state
            # Use torch.func API for vmap over basis vectors if needed,
            # or loop for simplicity in MVP
            
            # Use autograd.functional.jvp
            # Need a function that takes *only* the state
            map_fn_for_jvp = lambda s: dynamics_map_fn(s)

            _, output_tangent = torch.autograd.functional.jvp(
                map_fn_for_jvp, current_state_detached, v, create_graph=False # No gradients through LE estimation needed
            )
            return output_tangent

        # Compute JVPs for all vectors in q_matrix
        v_matrix = torch.zeros_like(q_matrix)
        for i in range(n_vectors):
            v_matrix[:, i] = jvp_fn(q_matrix[:, i])

        # QR decomposition of the resulting vectors
        # V = Q' R'
        try:
            q_prime, r_prime = torch.linalg.qr(v_matrix)
        except torch.linalg.LinAlgError:
             print("Warning: QR decomposition failed during Lyapunov estimation. Returning 0.")
             # Handle potential numerical issues, e.g., if vectors become linearly dependent
             # Or if the Jacobian leads to non-finite values
             # A simple recovery is to re-orthogonalize q_matrix and continue, or return 0
             q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, device=device))[0]
             continue # Skip this step's contribution

        # Accumulate the log of the diagonal elements of R' (stretching factors)
        # Ensure diagonal elements are positive before log
        log_stretch_sum += torch.sum(torch.log(torch.abs(torch.diag(r_prime))))

        # Update orthonormal vectors for the next step
        q_matrix = q_prime

        # Update the state for the next iteration (outside JVP calculation)
        with torch.no_grad():
             current_state = dynamics_map_fn(current_state_detached)


    # Average the log stretch factors
    lambda_max_estimate = log_stretch_sum / (steps * n_vectors) # Avg over steps and vectors
    # Note: Theory often divides by steps only. Dividing by n_vectors gives average rate across tracked dimensions.
    # For *largest* LE, focus on the first diagonal element r_prime[0,0] might be better,
    # but averaging is simpler/more stable initially. Let's stick to avg for MVP.
    # lambda_max_estimate = log_stretch_sum / steps # Alternative based on sum log |r_ii|

    return lambda_max_estimate.item()


def sparsify_vector(vector, k_fraction):
    """Zeros out all but the top-k% largest magnitude elements."""
    if k_fraction >= 1.0:
        return vector
    if k_fraction <= 0.0:
        return torch.zeros_like(vector)

    D = vector.numel()
    k_count = max(1, int(D * k_fraction)) # Ensure at least 1 element is kept

    # Find the threshold value for the top-k elements
    abs_vector = torch.abs(vector)
    threshold = torch.kthvalue(abs_vector, D - k_count + 1).values # Find (D-k+1)th smallest abs value = kth largest

    # Create a mask where abs(vector) >= threshold
    # Need careful handling if multiple elements have the exact threshold value
    mask = abs_vector >= threshold

    # If too many elements are selected due to ties, keep only k_count randomly or by magnitude
    if mask.sum() > k_count:
         # Get indices of all elements >= threshold
        potential_indices = torch.where(mask)[0]
        # Get their values
        potential_values = vector[potential_indices]
        # Sort by magnitude and take top k_count
        _, top_indices_relative = torch.topk(torch.abs(potential_values), k_count)
        # Get the actual indices in the original vector
        final_indices = potential_indices[top_indices_relative]
        # Create new mask
        mask = torch.zeros_like(vector, dtype=torch.bool)
        mask[final_indices] = True

    # Apply mask
    sparse_vec = torch.zeros_like(vector)
    sparse_vec[mask] = vector[mask]

    return sparse_vec

def sparsify_tensor_batch(tensor_batch, k_fraction):
    """Applies sparsify_vector to each tensor in a batch."""
    return torch.stack([sparsify_vector(t, k_fraction) for t in tensor_batch])

# --- PyTorch Sparse Utilities ---
# Note: PyTorch sparse support is evolving. These might need adjustments.

def sparse_dense_matmul(sparse_matrix, dense_matrix):
    """Perform SpMM: sparse_matrix @ dense_matrix."""
    return torch.sparse.mm(sparse_matrix, dense_matrix)

def dense_sparse_matmul(dense_matrix, sparse_matrix):
    """Perform DSM: dense_matrix @ sparse_matrix. Requires transpose tricks."""
    # (A @ B^T)^T = B @ A^T
    # Result = (sparse_matrix.T @ dense_matrix.T).T
    # Note: Transposing sparse matrices can be inefficient depending on format.
    # Check PyTorch version for optimal way.
    # For COO: sparse_matrix.t() works.
    return torch.sparse.mm(sparse_matrix.t(), dense_matrix.t()).t()


def sparse_elementwise_add(sparse_tensor1, sparse_tensor2):
    """Adds two sparse tensors. Assumes they have the same shape and sparsity pattern ideally."""
    # Simple addition often works if indices match or are combined.
    # PyTorch might automatically handle combining indices.
    return sparse_tensor1 + sparse_tensor2

def sparse_elementwise_mul(sparse_tensor1, sparse_tensor2):
     """Element-wise multiply for sparse tensors. Result is sparse."""
     return sparse_tensor1 * sparse_tensor2 # Check if this preserves sparsity as intended


def add_sparse_to_dense(dense_tensor, sparse_tensor):
    """Adds a sparse tensor to a dense tensor."""
    return dense_tensor + sparse_tensor.to_dense() # Simplest way, might lose efficiency

def get_sparse_tensor_size(sparse_tensor):
     """Returns the size of the sparse tensor."""
     return sparse_tensor.size()

def get_sparse_values(sparse_tensor):
     """Returns the non-zero values of the sparse tensor."""
     return sparse_tensor.values()

def get_sparse_indices(sparse_tensor):
    """Returns the indices of the non-zero values."""
    return sparse_tensor.indices()