# dpld/utils.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import time # For Logger timestamp

# --- Sparsification (Unchanged) ---
def sparsify_vector(dense_vector, k_sparse_factor):
    """Keeps top k% largest magnitude values, sets others to 0."""
    if not isinstance(dense_vector, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if dense_vector.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional.")
    if not (0 < k_sparse_factor <= 1):
        raise ValueError("k_sparse_factor must be between 0 (exclusive) and 1 (inclusive).")

    num_elements = dense_vector.numel()
    k = max(1, int(num_elements * k_sparse_factor)) # Ensure at least 1 element is kept

    top_k_indices = torch.topk(dense_vector.abs(), k).indices
    sparse_vector = torch.zeros_like(dense_vector)
    sparse_vector[top_k_indices] = dense_vector[top_k_indices]
    return sparse_vector

# --- Lyapunov Exponent Estimation (Unchanged from Rev 9) ---
def estimate_lyapunov_exponent(dynamics_map_func, initial_state_dense, n_vectors=5, steps=50, delta_t=1.0, device='cpu'):
    """
    Estimates the largest Lyapunov exponent using Jacobian-vector products.
    Args:
        dynamics_map_func: A function that takes a dense state tensor and returns the next dense state tensor.
        initial_state_dense: The starting state (dense tensor).
        n_vectors: Number of perturbation vectors to track.
        steps: Number of steps to iterate the dynamics and perturbations.
        delta_t: Time step size (usually 1.0 for discrete maps).
        device: PyTorch device.
    Returns:
        Estimated largest Lyapunov exponent (float), or None if unstable.
    """
    dim = initial_state_dense.numel()
    state = initial_state_dense.clone().detach().requires_grad_(False).to(device, dtype=torch.float64)

    q = torch.randn(dim, n_vectors, device=device, dtype=torch.float64)
    q, _ = torch.linalg.qr(q)

    log_stretch_sum = torch.zeros(n_vectors, device=device, dtype=torch.float64)

    for step in range(steps):
        state_input = state.clone().requires_grad_(True)
        next_state = dynamics_map_func(state_input)

        if not torch.all(torch.isfinite(next_state)):
            print(f"Warning: Non-finite value encountered in LE estimation dynamics map at step {step}. Aborting LE calc.")
            return None

        v = q.clone()
        jvp_results = []

        for i in range(n_vectors):
            vector = v[:, i].requires_grad_(False)
            try:
                next_state.backward(vector.to(next_state.dtype), retain_graph=True if i < n_vectors - 1 else False)
                jvp = state_input.grad.clone().detach()
                if jvp is None or not torch.all(torch.isfinite(jvp)):
                    print(f"Warning: Non-finite JVP for vector {i} at step {step}. Aborting LE calc.")
                    if state_input.grad is not None: state_input.grad.zero_()
                    return None
                jvp_results.append(jvp)
                if state_input.grad is not None: state_input.grad.zero_() # Ensure grad is zeroed after use
            except RuntimeError as e:
                 print(f"Error during JVP calculation for vector {i} at step {step}: {e}. Aborting LE calc.")
                 return None

        if len(jvp_results) != n_vectors:
            print(f"Warning: Incorrect number of JVP results ({len(jvp_results)} vs {n_vectors}). Aborting LE calc.")
            return None

        z = torch.stack(jvp_results, dim=1)

        try:
            q_new, r = torch.linalg.qr(z)
        except torch._C._LinAlgError as e:
            print(f"Warning: QR decomposition failed at step {step}: {e}. Aborting LE calc.")
            return None
        if not torch.all(torch.isfinite(r)):
            print(f"Warning: Non-finite values in R matrix at step {step}. Aborting LE calc.")
            return None

        diag_r = torch.diag(r)
        if torch.any(diag_r.abs() < 1e-10) or not torch.all(torch.isfinite(diag_r)):
             print(f"Warning: Invalid diagonal elements in R (zero, NaN, Inf) at step {step}: {diag_r}. Aborting LE calc.")
             return None

        log_stretch_sum += torch.log(diag_r.abs())
        q = q_new
        state = next_state.detach().requires_grad_(False).to(dtype=torch.float64)

    lyapunov_exponents = log_stretch_sum / (steps * delta_t)

    largest_le = torch.max(lyapunov_exponents).item()
    if not math.isfinite(largest_le):
        print(f"Warning: Final LE calculation resulted in non-finite value: {largest_le}")
        return None

    return largest_le


# --- Noise Schedule (Unchanged) ---
def linear_noise_decay(current_step, total_steps, start_noise, end_noise):
    """Linearly decays noise from start_noise to end_noise over total_steps."""
    if current_step >= total_steps:
        return end_noise
    fraction = current_step / total_steps
    return start_noise - fraction * (start_noise - end_noise)


# --- Logging (Revised Header for Rev 10) ---
class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.metrics = [] # Buffer for metrics
        self._create_log_file()

    def _create_log_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            # --- MODIFICATION Rev 10: Updated Header ---
            header = [
                "step", "Gt_log", "Gt_log_EMA", "Sm_log_avg", "Sm_log_std", "Sm_log_cls_avg",
                "Sm_log_cls_avg_generic", # Added
                "Sm_raw_cls_avg", "TaskHead_Sm_log_task", "TaskHead_Sm_raw_task",
                "lambda_max_est", "lambda_max_EMA", "gamma_t", "noise_std",
                "module_loss_avg", "module_policy_loss_avg", "module_pred_loss_avg",
                "meta_loss", "module_entropy_avg", "module_grad_norm_avg",
                "meta_grad_norm", "cls_norm", "cls_density",
                "TaskCorrect", "TaskAccuracy_Recent", # Added
                "OOD_Accuracy", "OOD_Loss"
            ]
            # --- END MODIFICATION ---
            with open(self.log_file, 'w') as f:
                f.write(','.join(header) + '\n')

    def log_metrics(self, metrics_dict, step):
        """Adds a dictionary of metrics for a given step to the buffer."""
        metrics_dict['step'] = step
        self.metrics.append(metrics_dict)

    def save_log(self):
        """Appends buffered metrics to the CSV log file."""
        if not self.metrics:
            return

        df = pd.DataFrame(self.metrics)
        try:
            header_cols = pd.read_csv(self.log_file, nrows=0).columns.tolist()
            df = df.reindex(columns=header_cols)
        except Exception as e:
            print(f"Warning: Could not read header from log file {self.log_file}. Saving with DataFrame columns. Error: {e}")

        df.to_csv(self.log_file, mode='a', header=False, index=False, na_rep='NaN')
        self.metrics = []


# --- Plotting (Revised for Rev 10 Metrics) ---
def plot_metrics(log_file_path):
    """Generates and saves plots from the metrics log file."""
    try:
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Log file is empty at {log_file_path}")
        return
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return

    log_dir = os.path.dirname(log_file_path)
    plot_file_path = os.path.join(log_dir, "metrics_plot.png")

    # MODIFIED Rev 10: Define specific columns to plot in order
    cols_to_plot = [
        "Gt_log", "Gt_log_EMA", "Sm_log_avg", "Sm_log_std", "Sm_log_cls_avg", "Sm_raw_cls_avg",
        "TaskHead_Sm_log_task", "TaskHead_Sm_raw_task", "TaskAccuracy_Recent", # Added Task Accuracy
        "lambda_max_EMA", "gamma_t", "noise_std", # Dynamics/Stability
        "module_loss_avg", "module_policy_loss_avg", "module_pred_loss_avg", "meta_loss", # Losses
        "module_entropy_avg", "module_grad_norm_avg", "meta_grad_norm", # Gradients/Entropy
        "cls_norm", "cls_density", # CLS stats
        "OOD_Accuracy", "OOD_Loss" # Eval
    ]
    # Filter out columns that don't exist or are all NaN
    cols_to_plot = [col for col in cols_to_plot if col in df.columns and not df[col].isnull().all() and df[col].dtype != object]

    num_metrics = len(cols_to_plot)
    if num_metrics == 0:
        print("No valid numeric metrics found in log file to plot.")
        return

    cols = 6 # Fixed number of columns for consistent layout
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for col in cols_to_plot:
        ax = axes[plot_idx]
        try:
            valid_data = df[[col, 'step']].dropna()
            if not valid_data.empty:
                ax.plot(valid_data['step'], valid_data[col], linewidth=0.8)
                ax.set_title(col)
                ax.set_xlabel("Step")
                ax.grid(True)
                # Specific y-axis limits for certain plots
                if col == "TaskAccuracy_Recent" or col == "OOD_Accuracy":
                    ax.set_ylim(-0.05, 1.05)
                elif col == "lambda_max_EMA":
                    # Find min/max excluding potential outliers if needed
                    min_val = valid_data[col].min()
                    max_val = valid_data[col].max()
                    padding = max(0.01, (max_val - min_val) * 0.1)
                    ax.set_ylim(min_val - padding, max_val + padding)

            else:
                ax.set_title(f"{col} (No Valid Data)")
                ax.text(0.5, 0.5, 'No Valid Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        except Exception as e:
            print(f"Error plotting column '{col}': {e}")
            ax.set_title(f"{col} (Plotting Error)")
            ax.text(0.5, 0.5, 'Plotting Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"DPLD Run Metrics ({os.path.basename(log_file_path)})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try:
        plt.savefig(plot_file_path)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving plot to {plot_file_path}: {e}")