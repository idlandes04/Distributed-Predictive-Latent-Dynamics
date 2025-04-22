# dpld/utils.py
# No changes needed in utils.py for this revision. Retain previous version.
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
import os
import time
import math

class Logger:
    """Simple logger for console and CSV (Unchanged from previous)."""
    def __init__(self, log_dir="logs", run_name=None):
        if run_name is None:
            run_name = f"dpld_run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.metrics = {}
        self.header_keys = None
        self.header_written = False
        self.last_log_time = time.time()
        print(f"Logging to {self.log_dir}")

    def log(self, step, metrics_dict):
        loggable_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
        self.metrics = {"step": step, **loggable_metrics}

        print(f"Step: {step}", end="")
        log_items_str = []
        sorted_keys = sorted(loggable_metrics.keys())
        for k in sorted_keys:
            v = loggable_metrics[k]
            if isinstance(v, torch.Tensor): v_item = v.item()
            elif isinstance(v, (np.generic, float, int)): v_item = v
            else: v_item = v # Keep other types as is for now

            if isinstance(v_item, float):
                 if math.isfinite(v_item): log_items_str.append(f"{k}: {v_item:.4f}")
                 else: log_items_str.append(f"{k}: {v_item}")
            else: log_items_str.append(f"{k}: {v_item}")
        print(" | ".join(log_items_str))


        if self.header_written:
            if self.header_keys is None: # Safety check
                 try:
                     with open(self.csv_path, 'r', newline='') as f: self.header_keys = next(csv.reader(f))
                 except Exception: self.header_keys = list(self.metrics.keys())
            # Ensure order matches header, handle missing keys
            data_row = [self.metrics.get(k, '') for k in self.header_keys]
        else:
            # Use sorted keys for the first write to ensure consistent order
            self.header_keys = ["step"] + sorted_keys
            data_row = [self.metrics.get(k, '') for k in self.header_keys]


        mode = 'a' if self.header_written else 'w'
        try:
            with open(self.csv_path, mode, newline='') as f:
                writer = csv.writer(f)
                if not self.header_written:
                    writer.writerow(self.header_keys)
                    self.header_written = True

                processed_data_row = []
                for item in data_row:
                    # Convert tensors/numpy types before checking finiteness
                    if isinstance(item, torch.Tensor): item_val = item.item()
                    elif isinstance(item, np.generic): item_val = item.item()
                    else: item_val = item

                    # Handle non-finite floats specifically for CSV writing
                    if isinstance(item_val, float) and not math.isfinite(item_val):
                         processed_data_row.append(str(item_val)) # Write 'nan' or 'inf' as string
                    else:
                         processed_data_row.append(item_val) # Write other types directly

                writer.writerow(processed_data_row)
        except IOError as e:
            print(f"Error writing to CSV {self.csv_path}: {e}")

    def close(self):
        pass

def plot_metrics(log_file, metrics_to_plot=None):
    """Plots metrics from a CSV log file (Unchanged from previous)."""
    import pandas as pd
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError: print(f"Error: Log file not found at {log_file}"); return
    except Exception as e: print(f"Error reading log file: {e}"); return
    if df.empty: print("Log file is empty."); return

    if metrics_to_plot is None:
        # Exclude non-numeric columns explicitly if needed, pandas usually handles this
        metrics_to_plot = [col for col in df.columns if col != 'step' and pd.api.types.is_numeric_dtype(df[col])]
        metrics_to_plot = [m for m in metrics_to_plot if m not in ['run_name']] # Exclude potential non-numeric

    num_plots = len(metrics_to_plot)
    if num_plots == 0: print("No numeric metrics found to plot."); return

    # Adjust layout dynamically
    cols = int(np.ceil(np.sqrt(num_plots)))
    if num_plots <= 3: cols = num_plots # Handle few plots better
    elif num_plots <= 8: cols = 4
    else: cols = 5 # Max 5 columns
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(min(25, 5 * cols), 4 * rows), squeeze=False)
    axes = axes.flatten()
    plot_count = 0
    for metric in metrics_to_plot:
        if metric in df.columns:
            # Convert column to numeric, coercing errors (like 'nan' strings)
            plot_data_series = pd.to_numeric(df[metric], errors='coerce')
            plot_data = pd.DataFrame({'step': df['step'], metric: plot_data_series}).dropna()

            if not plot_data.empty:
                axes[plot_count].plot(plot_data['step'], plot_data[metric])
                axes[plot_count].set_title(metric.replace('_', ' ').title())
                axes[plot_count].set_xlabel("Step"); axes[plot_count].grid(True)
                # Use scientific notation only if needed
                axes[plot_count].ticklabel_format(style='sci', axis='y', scilimits=(-3,4), useMathText=True)
                plot_count += 1
            # else: print(f"Warning: Metric '{metric}' has no plottable data after coercion.") # Less verbose
        # else: print(f"Warning: Metric '{metric}' not found.") # Less verbose

    for j in range(plot_count, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    run_dir_name = os.path.basename(os.path.dirname(log_file))
    plt.suptitle(f"DPLD Run Metrics ({run_dir_name})", fontsize=16)
    plot_filename = os.path.splitext(log_file)[0] + "_plot.png"
    try: plt.savefig(plot_filename); print(f"Metrics plot saved to {plot_filename}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close(fig)


def estimate_lyapunov_exponent(dynamics_map_fn, state, n_vectors=5, steps=50, epsilon=1e-5, device='cpu'):
    """
    Estimates the largest Lyapunov exponent using Jacobian-vector products.
    Includes enhanced robustness checks and fix for tanh input type.
    """
    with torch.no_grad():
        D = state.shape[0]
        if D < n_vectors: n_vectors = D
        if n_vectors == 0: return None

        try:
            # Initialize Q matrix with float64 for precision
            q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, dtype=torch.float64, device=device))[0]
        except Exception as e:
             print(f"Error: LE QR initialization failed: {e}"); return None

        log_stretch_sum = torch.tensor(0.0, dtype=torch.float64, device=device) # Use float64 for sum
        current_state = state.detach().clone().to(torch.float64) # Use float64 for state evolution internally
        if not torch.all(torch.isfinite(current_state)):
             print("Error: Initial state for LE is non-finite."); return None

    def jvp_fn(v_in):
        current_state_detached = current_state.detach().requires_grad_(True)
        # Ensure the dynamics map handles float64 input and returns float64 output
        map_fn_for_jvp = lambda s: dynamics_map_fn(s.to(state.dtype)).to(torch.float64)
        try:
            v_in_casted = v_in.to(current_state_detached.dtype) # JVP input tangent vector matches state dtype
            _, output_tangent = torch.autograd.functional.jvp(
                map_fn_for_jvp, current_state_detached, v_in_casted, create_graph=False
            )
            if not torch.all(torch.isfinite(output_tangent)):
                # print("Warning: Non-finite values detected in JVP output.") # Less verbose
                return None
            return output_tangent
        except Exception as e:
             # print(f"Error during JVP calculation: {e}") # Less verbose
             return None

    valid_steps = 0
    for step_num in range(steps):
        v_list = []
        jvp_failed = False
        for i in range(n_vectors):
            v_out = jvp_fn(q_matrix[:, i]) # q_matrix is float64, v_in_casted inside jvp_fn handles dtype
            if v_out is None: jvp_failed = True; break
            v_list.append(v_out)

        if jvp_failed:
            # Attempt to recover by re-orthogonalizing and advancing state
            # print(f"LE Step {step_num}: JVP failed. Skipping & re-orthogonalizing.") # Less verbose
            try: q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, dtype=torch.float64, device=device))[0]
            except Exception: print("LE: Recovery QR failed."); return None
            with torch.no_grad():
                 next_state_candidate = dynamics_map_fn(current_state.detach().to(state.dtype)).to(torch.float64)
                 if not torch.all(torch.isfinite(next_state_candidate)): print("LE: State non-finite during recovery."); return None
                 current_state = next_state_candidate
            continue

        if not v_list: continue # Should not happen if jvp_failed didn't trigger
        v_matrix = torch.stack(v_list, dim=1) # Stack the float64 tangent vectors

        if not torch.all(torch.isfinite(v_matrix)): # Check before QR
            # print(f"Warning: Non-finite JVP result matrix at LE step {step_num}. Skipping & re-orthogonalizing.") # Less verbose
            try: q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, dtype=torch.float64, device=device))[0]
            except Exception: print("LE: Recovery QR failed."); return None
            with torch.no_grad():
                 next_state_candidate = dynamics_map_fn(current_state.detach().to(state.dtype)).to(torch.float64)
                 if not torch.all(torch.isfinite(next_state_candidate)): print("LE: State non-finite during recovery."); return None
                 current_state = next_state_candidate
            continue

        try:
            # QR decomposition on the float64 matrix
            q_prime, r_prime = torch.linalg.qr(v_matrix)
        except torch.linalg.LinAlgError as e:
             # print(f"Warning: QR decomposition failed during LE step {step_num}: {e}. Skipping & re-orthogonalizing.") # Less verbose
             try: q_matrix = torch.linalg.qr(torch.randn(D, n_vectors, dtype=torch.float64, device=device))[0]
             except Exception: print("LE: Recovery QR failed."); return None
             with torch.no_grad():
                  next_state_candidate = dynamics_map_fn(current_state.detach().to(state.dtype)).to(torch.float64)
                  if not torch.all(torch.isfinite(next_state_candidate)): print("LE: State non-finite during recovery."); return None
                  current_state = next_state_candidate
             continue

        diag_r = torch.diag(r_prime)
        # Use abs() before log, clamp positive values away from zero
        safe_diag_r = torch.clamp(torch.abs(diag_r), min=1e-12) # Clamp closer to zero for float64
        log_stretch_sum += torch.sum(torch.log(safe_diag_r))
        valid_steps += 1
        q_matrix = q_prime # Update Q matrix

        # Advance the state using the dynamics map
        with torch.no_grad():
             next_state_candidate = dynamics_map_fn(current_state.detach().to(state.dtype)).to(torch.float64)
             if not torch.all(torch.isfinite(next_state_candidate)):
                  # print(f"Error: Non-finite state during LE update step {step_num}.") # Less verbose
                  return None
             current_state = next_state_candidate

    if valid_steps == 0: return None
    # Average log stretch over valid steps
    lambda_max_estimate = log_stretch_sum / valid_steps
    final_estimate = lambda_max_estimate.item()
    # Final check for sanity
    if not math.isfinite(final_estimate): return None
    return final_estimate


def sparsify_vector(vector, k_fraction):
    """Zeros out all but the top-k% largest magnitude elements (Unchanged)."""
    if not isinstance(vector, torch.Tensor): raise TypeError("Input must be a PyTorch tensor.")
    if vector.dim() != 1: raise ValueError("Input tensor must be 1-dimensional.")
    if k_fraction >= 1.0: return vector
    if k_fraction <= 0.0: return torch.zeros_like(vector)
    D = vector.numel()
    if D == 0: return vector
    k_count = max(1, min(D, int(D * k_fraction)))
    abs_vector = torch.abs(vector)
    # Handle potential non-finite values before topk
    valid_abs_vector = torch.nan_to_num(abs_vector, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf'))
    # Clamp large values to prevent issues with topk on some platforms/dtypes
    valid_abs_vector = torch.clamp(valid_abs_vector, max=torch.finfo(vector.dtype).max / 2)
    try:
        actual_k = min(k_count, D)
        if actual_k == 0: return torch.zeros_like(vector)
        # Find the k-th largest absolute value
        top_values, top_indices = torch.topk(valid_abs_vector, actual_k)
        threshold = top_values[-1] if actual_k > 0 else -float('inf')
    except RuntimeError as e:
         # print(f"Warning: Error in topk during sparsify ({e}). Keeping all elements.") # Less verbose
         return vector # Return original vector if topk fails

    # Create mask based on threshold
    mask = abs_vector >= threshold
    # Refine mask if too many elements meet the threshold due to ties
    if mask.sum() > actual_k:
         # Fallback to using the indices directly if ties cause issues
         mask = torch.zeros_like(vector, dtype=torch.bool, device=vector.device)
         mask[top_indices] = True
    elif mask.sum() < actual_k and actual_k > 0: # Ensure exactly k are selected if possible
         mask = torch.zeros_like(vector, dtype=torch.bool, device=vector.device)
         mask[top_indices] = True

    sparse_vec = torch.zeros_like(vector)
    if mask.sum() > 0: sparse_vec[mask] = vector[mask] # Apply mask
    return sparse_vec