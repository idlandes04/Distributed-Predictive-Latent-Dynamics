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

from core import DPLDSystem, DEFAULT_ACTION_STD, DEFAULT_PREDICTION_LOSS_WEIGHT, DEFAULT_TASK_LOSS_WEIGHT
from envs import ArithmeticEnv
from utils import Logger, plot_metrics, linear_noise_decay

def evaluate_ood(system, env, num_eval_steps=1000):
    """Evaluates TaskHead performance on Out-of-Distribution data."""
    system.eval()
    total_correct = 0
    total_loss = 0
    env.set_range('ood', min_val=env.train_max_val + 1, max_val=env.train_max_val + 100)

    with torch.no_grad():
        # Reset TaskHead hidden state for evaluation consistency
        system.task_head.last_gru_hidden_state = None
        current_cls_state = system.ct # Use the current CLS state at eval time

        for _ in range(num_eval_steps):
            task_input = env.step()
            a, op_idx, b, true_answer_c = task_input

            # Simulate prediction based on current CLS state
            try:
                # Predict using the TaskHead's internal model
                _ = system.task_head.predict(current_cls_state) # Updates internal state/prediction
                predicted_answer = system.task_head.get_task_prediction() # Get detached task prediction
            except Exception as e:
                print(f"Warning: Error during OOD prediction: {e}")
                predicted_answer = None

            if predicted_answer is not None and true_answer_c is not None and torch.isfinite(predicted_answer) and torch.isfinite(true_answer_c):
                # Accuracy check
                if abs(predicted_answer.item() - true_answer_c.item()) < 0.5:
                    total_correct += 1
                # Calculate MSE loss
                loss = F.mse_loss(predicted_answer, true_answer_c.float().to(predicted_answer.device))
                if math.isfinite(loss.item()):
                    total_loss += loss.item()
                else:
                    total_loss += 1e6 # Penalize non-finite loss
            else:
                 total_loss += 1e6 # Penalize no prediction or NaN/Inf

    avg_accuracy = total_correct / num_eval_steps
    avg_loss = total_loss / num_eval_steps
    env.set_range('train') # Switch back to training mode
    system.train()
    return avg_accuracy, avg_loss


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = args.log_dir or f"logs_p1_rev9_{time.strftime('%Y%m%d_%H%M%S')}" # Updated log name
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
        k_sparse_write=args.k_sparse,
        module_lr=args.module_lr,
        meta_lr=args.meta_lr,
        taskhead_lr=args.taskhead_lr, # MODIFIED Rev 9: Pass taskhead_lr
        noise_std_dev_schedule=noise_schedule,
        env=env,
        embedding_dim=args.embedding_dim,
        entropy_coeff=args.entropy_coeff,
        ema_alpha=args.ema_alpha,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        stability_target=args.stability_target,
        action_std=args.action_std,
        prediction_loss_weight=args.prediction_loss_weight,
        task_loss_weight=args.task_loss_weight,
        weight_decay=args.weight_decay,
        clip_cls_norm=not args.no_cls_clip,
        device=device
    ).to(device)

    logger = Logger(log_dir=log_dir)

    # --- Training Loop ---
    start_time = time.time()
    pbar = tqdm(range(args.total_steps))
    last_log_save_step = -1 # Track step when log was last saved
    metrics = {} # Initialize metrics dict

    for step in pbar:
        estimate_le = (args.le_interval > 0 and step > 0 and step % args.le_interval == 0)

        task_input = env.step()
        true_answer_c = task_input[3]

        try:
            metrics = system.step(step, task_input, estimate_le=estimate_le, true_answer_c=true_answer_c)

            # MODIFIED Rev 9: Log metrics to buffer only periodically
            if step % args.log_save_interval == 0:
                # Add OOD metrics if they were calculated in this interval
                if 'OOD_Accuracy' in metrics: # Check if OOD keys exist from eval
                    logger.log_metrics(metrics, step)
                else:
                    # Fetch latest OOD metrics if available from previous eval
                    latest_ood_acc = metrics.get("OOD_Accuracy", float('nan'))
                    latest_ood_loss = metrics.get("OOD_Loss", float('nan'))
                    metrics_to_log = metrics.copy()
                    metrics_to_log["OOD_Accuracy"] = latest_ood_acc
                    metrics_to_log["OOD_Loss"] = latest_ood_loss
                    logger.log_metrics(metrics_to_log, step)

            critical_keys = ['Gt_log', 'cls_norm', 'module_loss_avg', 'module_policy_loss_avg', 'module_pred_loss_avg', 'meta_loss']
            if any(k in metrics and (metrics[k] is None or not math.isfinite(metrics[k])) for k in critical_keys):
                 print(f"\nNaN detected in critical metrics at step {step}. Aborting.")
                 log_subset = {k: metrics.get(k, 'N/A') for k in ['step'] + critical_keys + ['module_grad_norm_avg']}
                 print("Last metrics:", log_subset)
                 break

        except Exception as e:
            print(f"\nError during system step {step}: {e}")
            traceback.print_exc()
            break

        # --- Logging & Saving ---
        # Update progress bar description periodically
        if step % args.pbar_update_freq == 0 or step == args.total_steps - 1:
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed_time if elapsed_time > 0 else 0
            total_estimated_time = elapsed_time / (step + 1) * args.total_steps if step > 0 else 0
            eta_seconds = total_estimated_time - elapsed_time if total_estimated_time > elapsed_time else 0

            # Use the latest metrics dictionary for the progress bar
            pbar.set_postfix({
                "Gt_log": f"{metrics.get('Gt_log', float('nan')):.2f}",
                "TaskSm": f"{metrics.get('TaskHead_Sm_log_task', float('nan')):.2f}",
                "ModLoss": f"{metrics.get('module_loss_avg', float('nan')):.2f}",
                "ModGrad": f"{metrics.get('module_grad_norm_avg', float('nan')):.2f}",
                "CLS Norm": f"{metrics.get('cls_norm', float('nan')):.1f}",
                "Gamma": f"{metrics.get('gamma_t', float('nan')):.3f}",
                "LE_EMA": f"{metrics.get('lambda_max_EMA', float('nan')):.2f}",
                "Steps/s": f"{steps_per_sec:.1f}",
                "ETA": f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            })

        # Save log file periodically based on step interval
        if step > 0 and step % args.log_save_interval == 0:
             logger.save_log() # Save buffered metrics
             last_log_save_step = step

        # --- OOD Evaluation ---
        if args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
             ood_acc, ood_loss = evaluate_ood(system, env, args.eval_steps)
             # Store OOD results in the metrics dict for the *next* logging interval
             metrics["OOD_Accuracy"] = ood_acc
             metrics["OOD_Loss"] = ood_loss
             tqdm.write(f"Step {step}: OOD Accuracy: {ood_acc:.4f}, OOD Loss: {ood_loss:.4f}")


    # --- Save final state and plot ---
    # Ensure any remaining metrics in the buffer are saved
    if last_log_save_step < args.total_steps -1:
        logger.save_log()

    final_checkpoint_path = os.path.join(log_dir, "final_model_state.pth")
    torch.save(system.state_dict(), final_checkpoint_path)
    print(f"Final model state saved to {final_checkpoint_path}")
    try:
        plot_metrics(logger.log_file)
        print(f"Metrics plot saved to {os.path.join(log_dir, 'metrics_plot.png')}")
    except Exception as e:
        print(f"Failed to generate metrics plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD Model - Revision 9")
    # Model Params
    parser.add_argument('--cls-dim', type=int, default=4096, help='Dimension of Central Latent Space')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of generic predictive modules')
    parser.add_argument('--module-hidden-dim', type=int, default=128, help='Hidden dimension for predictive modules')
    parser.add_argument('--meta-hidden-dim', type=int, default=64, help='Hidden dimension for meta-model')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Dimension for task input embeddings')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity factor for CLS writes')
    # Training Params
    parser.add_argument('--total-steps', type=int, default=20000, help='Total number of training steps') # Default for Rev 9
    parser.add_argument('--module-lr', type=float, default=5e-5, help='Learning rate for generic predictive modules')
    parser.add_argument('--taskhead-lr', type=float, default=1e-5, help='Learning rate for TaskHead module') # MODIFIED Rev 9: Added
    parser.add_argument('--meta-lr', type=float, default=5e-5, help='Learning rate for meta-model')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for Adam optimizers')
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Coefficient for entropy bonus')
    parser.add_argument('--action-std', type=float, default=DEFAULT_ACTION_STD, help='Standard deviation for module action sampling') # Default changed in core.py
    parser.add_argument('--prediction-loss-weight', type=float, default=DEFAULT_PREDICTION_LOSS_WEIGHT, help='Weight for CLS prediction loss component')
    parser.add_argument('--task-loss-weight', type=float, default=DEFAULT_TASK_LOSS_WEIGHT, help='Weight for Task prediction loss component (TaskHead only)')
    # Dynamics Params
    parser.add_argument('--gamma-min', type=float, default=0.01, help='Min value for CLS decay rate gamma_t')
    parser.add_argument('--gamma-max', type=float, default=0.2, help='Max value for CLS decay rate gamma_t')
    parser.add_argument('--noise-start', type=float, default=0.05, help='Initial noise standard deviation')
    parser.add_argument('--noise-end', type=float, default=0.001, help='Final noise standard deviation')
    parser.add_argument('--ema-alpha', type=float, default=0.99, help='EMA decay factor for smoothing Gt and lambda_max')
    parser.add_argument('--no-cls-clip', action='store_true', help='Disable CLS norm clipping')
    # Stability Params
    parser.add_argument('--le-interval', type=int, default=1000, help='Interval for estimating Lyapunov exponent (-1 to disable)') # Default for Rev 9
    parser.add_argument('--stability-target', type=float, default=-0.01, help='Target value for lambda_max_EMA for Meta-Model reward')
    # Environment Params
    parser.add_argument('--max-val', type=int, default=100, help='Maximum value for numbers in arithmetic task')
    # Logging/Eval Params
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs and checkpoints (default: logs_p1_rev9_YYMMDD_HHMMSS)')
    parser.add_argument('--pbar-update-freq', type=int, default=10, help='Update progress bar postfix every N steps')
    parser.add_argument('--log-save-interval', type=int, default=200, help='Buffer metrics and save log file every N steps') # MODIFIED Rev 9: Default
    parser.add_argument('--eval-interval', type=int, default=2000, help='Evaluate OOD performance every N steps (-1 to disable)')
    parser.add_argument('--eval-steps', type=int, default=500, help='Number of steps for OOD evaluation')
    # System Params
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    main(args)