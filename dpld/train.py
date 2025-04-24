# dpld/train.py

import torch
import torch.nn.functional as F
import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import traceback # For detailed error printing

# MODIFIED Rev 11: Adjusted defaults, added module_write_scale
from core import DPLDSystem, DEFAULT_ACTION_STD # Removed unused loss weight defaults
from envs import ArithmeticEnv
from utils import Logger, plot_metrics, linear_noise_decay

def evaluate_ood(system, env, num_eval_steps=1000):
    """Evaluates TaskHead performance on Out-of-Distribution data."""
    system.eval()
    total_correct = 0
    total_loss = 0
    env.set_range('ood', min_val=env.train_max_val + 1, max_val=env.train_max_val + 100)

    with torch.no_grad():
        # Reset TaskHead state if applicable (MLP has no state, GRU would need reset)
        # system.task_head.last_gru_hidden_state = None # Not needed for MLP TaskHead
        current_cls_state = system.ct # Use current CLS state

        for _ in range(num_eval_steps):
            task_input = env.step()
            a, op_idx, b, true_answer_c = task_input
            try:
                # Predict using the simplified TaskHead
                _ = system.task_head.predict(current_cls_state) # Reads fixed indices
                predicted_answer = system.task_head.get_task_prediction() # Gets MLP output
            except Exception as e:
                print(f"Warning: Error during OOD prediction: {e}")
                predicted_answer = None

            if predicted_answer is not None and true_answer_c is not None and torch.isfinite(predicted_answer) and torch.isfinite(true_answer_c):
                # Check accuracy
                if abs(predicted_answer.item() - true_answer_c.item()) < 0.5:
                    total_correct += 1
                # Calculate loss
                loss = F.mse_loss(predicted_answer, true_answer_c.float().to(predicted_answer.device))
                if math.isfinite(loss.item()):
                    total_loss += loss.item()
                else:
                    total_loss += 1e6 # Penalize non-finite loss
            else:
                 total_loss += 1e6 # Penalize invalid prediction/answer

    avg_accuracy = total_correct / num_eval_steps
    avg_loss = total_loss / num_eval_steps
    env.set_range('train') # Switch back to training range
    system.train() # Set system back to training mode
    return avg_accuracy, avg_loss


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = args.log_dir or f"logs_p1_rev11_{time.strftime('%Y%m%d_%H%M%S')}" # Updated log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Logging to: {log_dir}")

    env = ArithmeticEnv(max_val=args.max_val, device=device)
    noise_schedule = lambda step: linear_noise_decay(step, args.total_steps, args.noise_start, args.noise_end)

    system = DPLDSystem(
        cls_dim=args.cls_dim,
        num_modules=args.num_modules,
        module_hidden_dim=args.module_hidden_dim,
        meta_hidden_dim=args.meta_hidden_dim,
        k_sparse_write=args.k_sparse, # Used by generic modules
        module_lr=args.module_lr,
        meta_lr=args.meta_lr,
        taskhead_lr=args.taskhead_lr, # Used by simplified TaskHead MLP
        noise_std_dev_schedule=noise_schedule,
        env=env,
        embedding_dim=args.embedding_dim, # Unused by MathEncoder now
        entropy_coeff=args.entropy_coeff, # Used by generic modules
        ema_alpha=args.ema_alpha,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        stability_target=args.stability_target,
        action_std=args.action_std, # Used by generic modules
        # prediction_loss_weight=args.prediction_loss_weight, # No longer used by TaskHead
        # task_loss_weight=args.task_loss_weight, # No longer used by TaskHead
        module_write_scale=args.module_write_scale, # Added Rev 11
        weight_decay=args.weight_decay,
        clip_cls_norm=not args.no_cls_clip,
        device=device
    ).to(device)

    logger = Logger(log_dir=log_dir)

    # --- Training Loop ---
    start_time = time.time()
    pbar = tqdm(range(args.total_steps))
    last_log_save_step = -1
    metrics = {}
    task_correct_buffer = [] # Accumulator for recent task accuracy

    for step in pbar:
        # Determine if LE should be estimated this step
        estimate_le = (args.le_interval > 0 and step > 0 and step % args.le_interval == 0)

        # Get task input from environment
        task_input = env.step()
        true_answer_c = task_input[3] # Extract true answer

        try:
            # Perform one step of the DPLD system
            metrics = system.step(step, task_input, estimate_le=estimate_le, true_answer_c=true_answer_c)

            # Accumulate task correctness flag
            task_correct_buffer.append(metrics.get("TaskCorrect", 0))

            # Log metrics to buffer periodically
            if step > 0 and step % args.log_save_interval == 0:
                # Calculate recent task accuracy from buffer
                if task_correct_buffer:
                    recent_acc = np.mean(task_correct_buffer)
                    metrics["TaskAccuracy_Recent"] = recent_acc
                    task_correct_buffer = [] # Reset buffer after logging
                else:
                    metrics["TaskAccuracy_Recent"] = float('nan') # Handle empty buffer case

                # Fetch latest OOD metrics if they exist in the current metrics dict
                latest_ood_acc = metrics.get("OOD_Accuracy", float('nan'))
                latest_ood_loss = metrics.get("OOD_Loss", float('nan'))

                # Prepare metrics dictionary for logging (ensure all expected columns are present)
                metrics_to_log = metrics.copy()
                metrics_to_log["OOD_Accuracy"] = latest_ood_acc
                metrics_to_log["OOD_Loss"] = latest_ood_loss

                logger.log_metrics(metrics_to_log, step) # Log buffered metrics

            # Check for critical NaN/Inf values
            critical_keys = ['Gt_log', 'cls_norm', 'module_loss_avg', 'meta_loss'] # Removed policy/pred loss averages as TaskHead changed
            if any(k in metrics and (metrics[k] is None or not math.isfinite(metrics[k])) for k in critical_keys):
                 print(f"\nNaN/Inf detected in critical metrics at step {step}. Aborting.")
                 log_subset = {k: metrics.get(k, 'N/A') for k in ['step'] + critical_keys + ['module_grad_norm_avg']}
                 print("Last metrics:", log_subset)
                 break # Stop training

        except Exception as e:
            print(f"\nError during system step {step}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            break # Stop training on error

        # --- Update Progress Bar ---
        if step % args.pbar_update_freq == 0 or step == args.total_steps - 1:
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed_time if elapsed_time > 0 else 0
            total_estimated_time = elapsed_time / (step + 1) * args.total_steps if step > 0 else 0
            eta_seconds = total_estimated_time - elapsed_time if total_estimated_time > elapsed_time else 0

            pbar.set_postfix({
                "Gt_log": f"{metrics.get('Gt_log', float('nan')):.3f}",
                "TaskSm": f"{metrics.get('TaskHead_Sm_log_task', float('nan')):.2f}",
                "TaskAcc": f"{metrics.get('TaskAccuracy_Recent', float('nan')):.2f}",
                "ModLoss": f"{metrics.get('module_loss_avg', float('nan')):.2f}", # Avg loss of all modules
                "ModGrad": f"{metrics.get('module_grad_norm_avg', float('nan')):.1f}",
                "CLS Norm": f"{metrics.get('cls_norm', float('nan')):.1f}",
                "Gamma": f"{metrics.get('gamma_t', float('nan')):.3f}",
                "LE_EMA": f"{metrics.get('lambda_max_EMA', float('nan')):.3f}", # Use EMA for display
                "Steps/s": f"{steps_per_sec:.1f}",
                "ETA": f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            })

        # --- Save Log File Periodically ---
        if step > 0 and step % args.log_save_interval == 0:
             logger.save_log() # Write buffered metrics to file
             last_log_save_step = step

        # --- Evaluate OOD Performance Periodically ---
        if args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
             ood_acc, ood_loss = evaluate_ood(system, env, args.eval_steps)
             # Store OOD results in the main metrics dict to be picked up by the next log save
             metrics["OOD_Accuracy"] = ood_acc
             metrics["OOD_Loss"] = ood_loss
             # Print immediately for feedback
             tqdm.write(f"Step {step}: OOD Accuracy: {ood_acc:.4f}, OOD Loss: {ood_loss:.4f}")


    # --- Final Save and Plot ---
    # Save any remaining metrics if the loop finished before a save interval
    if last_log_save_step < args.total_steps -1:
        if task_correct_buffer: metrics["TaskAccuracy_Recent"] = np.mean(task_correct_buffer)
        logger.log_metrics(metrics, args.total_steps - 1) # Log final metrics
        logger.save_log() # Save the log file one last time

    # Save the final model state
    final_checkpoint_path = os.path.join(log_dir, "final_model_state.pth")
    torch.save(system.state_dict(), final_checkpoint_path)
    print(f"Final model state saved to {final_checkpoint_path}")

    # Generate metrics plot
    try:
        plot_metrics(logger.log_file)
        print(f"Metrics plot saved to {os.path.join(log_dir, 'metrics_plot.png')}")
    except Exception as e:
        print(f"Failed to generate metrics plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD Model - Revision 11")
    # Model Params
    parser.add_argument('--cls-dim', type=int, default=4096, help='Dimension of Central Latent Space')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of generic predictive modules')
    parser.add_argument('--module-hidden-dim', type=int, default=128, help='Hidden dimension for modules')
    parser.add_argument('--meta-hidden-dim', type=int, default=64, help='Hidden dimension for meta-model')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dim (unused by MathEncoder Rev11)')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity factor for generic module CLS writes')
    # Training Params
    parser.add_argument('--total-steps', type=int, default=50000, help='Total number of training steps')
    parser.add_argument('--module-lr', type=float, default=5e-5, help='Learning rate for generic predictive modules')
    parser.add_argument('--taskhead-lr', type=float, default=1e-4, help='Learning rate for TaskHead MLP (Rev11)') # Increased for simpler model
    parser.add_argument('--meta-lr', type=float, default=5e-5, help='Learning rate for meta-model')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for Adam optimizers')
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Coefficient for entropy bonus (generic modules)')
    parser.add_argument('--action-std', type=float, default=DEFAULT_ACTION_STD, help='Std dev for generic module action sampling')
    # parser.add_argument('--prediction-loss-weight', type=float, default=1.0, help='Weight for CLS prediction loss (generic modules)') # Simplified, use 1.0
    # parser.add_argument('--task-loss-weight', type=float, default=1.0, help='Weight for Task prediction loss (TaskHead only)') # Simplified, use 1.0
    parser.add_argument('--module-write-scale', type=float, default=0.5, help='Scaling factor for generic module write vectors (Rev11)') # Added
    # Dynamics Params
    parser.add_argument('--gamma-min', type=float, default=0.02, help='Min value for CLS decay rate gamma_t')
    parser.add_argument('--gamma-max', type=float, default=0.2, help='Max value for CLS decay rate gamma_t')
    parser.add_argument('--noise-start', type=float, default=0.05, help='Initial noise standard deviation')
    parser.add_argument('--noise-end', type=float, default=0.001, help='Final noise standard deviation')
    parser.add_argument('--ema-alpha', type=float, default=0.99, help='EMA decay factor for smoothing Gt and lambda_max')
    parser.add_argument('--no-cls-clip', action='store_true', help='Disable CLS norm clipping')
    # Stability Params
    parser.add_argument('--le-interval', type=int, default=1000, help='Interval for estimating Lyapunov exponent (-1 to disable)')
    parser.add_argument('--stability-target', type=float, default=-0.01, help='Target value for lambda_max_EMA for Meta-Model reward')
    # Environment Params
    parser.add_argument('--max-val', type=int, default=100, help='Maximum value for numbers in arithmetic task')
    # Logging/Eval Params
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs and checkpoints (default: logs_p1_rev11_YYMMDD_HHMMSS)')
    parser.add_argument('--pbar-update-freq', type=int, default=10, help='Update progress bar postfix every N steps')
    parser.add_argument('--log-save-interval', type=int, default=200, help='Buffer metrics and save log file every N steps')
    parser.add_argument('--eval-interval', type=int, default=2500, help='Evaluate OOD performance every N steps (-1 to disable)')
    parser.add_argument('--eval-steps', type=int, default=500, help='Number of steps for OOD evaluation')
    # System Params
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    main(args)