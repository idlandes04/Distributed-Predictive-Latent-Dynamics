# dpld/train.py

import torch
import torch.nn.functional as F # Added for evaluate_ood loss
import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import math # Added for isnan/isfinite checks

# --- MODIFICATION: Import DEFAULT_ACTION_STD and DEFAULT_PREDICTION_LOSS_WEIGHT ---
from core import DPLDSystem, DEFAULT_ACTION_STD, DEFAULT_PREDICTION_LOSS_WEIGHT
# --- END MODIFICATION ---

from envs import ArithmeticEnv
from utils import Logger, plot_metrics, linear_noise_decay

def evaluate_ood(system, env, num_eval_steps=1000):
    """Evaluates TaskHead performance on Out-of-Distribution data."""
    system.eval() # Set system to evaluation mode
    total_correct = 0
    total_loss = 0
    # --- MODIFICATION: Use env methods consistently ---
    env.set_range('ood', min_val=env.train_max_val + 1, max_val=env.train_max_val + 100) # Example OOD range
    # --- END MODIFICATION ---

    with torch.no_grad():
        for _ in range(num_eval_steps):
            # --- MODIFICATION: Use env methods consistently ---
            task_input = env.step() # Get next problem tuple
            # --- END MODIFICATION ---
            a, op_idx, b, true_answer_c = task_input

            # --- MODIFICATION: Simulate step for prediction only ---
            # We need to call predict on the TaskHead to get its output
            # This requires the current CLS state, but we don't need to run the full step
            # For simplicity in eval, let's assume we can just get the prediction
            # based on the *current* state, acknowledging this isn't strictly the
            # next-step prediction the model is trained on.
            # A more accurate eval would run a non-learning step.
            # Let's run a simplified step:
            try:
                _ = system.task_head.predict(system.ct) # Update internal state/prediction
                predicted_answer = system.task_head.get_task_prediction() # Get task prediction (detached)
            except Exception:
                predicted_answer = None # Handle potential errors during prediction

            # --- END MODIFICATION ---

            if predicted_answer is not None and true_answer_c is not None and torch.isfinite(predicted_answer):
                # Simple accuracy: check if prediction is close to true answer
                if abs(predicted_answer.item() - true_answer_c.item()) < 0.5: # Threshold for correctness
                    total_correct += 1
                # Calculate MSE loss for logging
                loss = F.mse_loss(predicted_answer, true_answer_c.float().to(predicted_answer.device))
                total_loss += loss.item()
            else:
                 # Penalize heavily if no prediction or NaN/Inf
                 total_loss += 1e6

    avg_accuracy = total_correct / num_eval_steps
    avg_loss = total_loss / num_eval_steps
    # --- MODIFICATION: Use env methods consistently ---
    env.set_range('train') # Switch env back to training mode
    # --- END MODIFICATION ---
    system.train() # Set system back to training mode
    return avg_accuracy, avg_loss


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

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
        noise_std_dev_schedule=noise_schedule,
        env=env,
        embedding_dim=args.embedding_dim,
        entropy_coeff=args.entropy_coeff,
        ema_alpha=args.ema_alpha,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        stability_target=args.stability_target, # Pass stability target
        action_std=args.action_std, # Pass action std
        prediction_loss_weight=args.prediction_loss_weight, # Pass prediction loss weight
        task_loss_weight=0.0, # Force task weight to 0 for this run
        weight_decay=args.weight_decay, # Pass weight decay
        clip_cls_norm=not args.no_cls_clip,
        device=device
    ).to(device)

    logger = Logger(log_dir=args.log_dir)

    # --- Training Loop ---
    start_time = time.time()
    pbar = tqdm(range(args.total_steps))
    last_log_time = start_time
    steps_since_log = 0

    for step in pbar:
        # --- MODIFICATION: Enable LE estimation again ---
        estimate_le = (args.le_interval > 0 and step > 0 and step % args.le_interval == 0) # Don't run at step 0
        # --- END MODIFICATION ---

        # --- MODIFICATION: Use env methods consistently ---
        task_input = env.step() # Get next problem tuple
        # --- END MODIFICATION ---
        true_answer_c = task_input[3] # Get true answer for potential later use

        try:
            metrics = system.step(step, task_input, estimate_le=estimate_le, true_answer_c=true_answer_c)
            # --- MODIFICATION: Log new loss components ---
            logger.log_metrics(metrics, step) # Logger needs to handle new keys if added
            # --- END MODIFICATION ---

            # Check for NaNs in critical metrics
            critical_keys = ['Gt_log', 'cls_norm', 'module_loss_avg', 'module_policy_loss_avg', 'module_pred_loss_avg', 'meta_loss']
            if any(k in metrics and (metrics[k] is None or not math.isfinite(metrics[k])) for k in critical_keys):
                 print(f"\nNaN detected in critical metrics at step {step}. Aborting.")
                 log_subset = {k: metrics.get(k, 'N/A') for k in ['step'] + critical_keys + ['module_grad_norm_avg']}
                 print("Last metrics:", log_subset)
                 break

        except Exception as e:
            print(f"\nError during system step {step}: {e}")
            import traceback
            traceback.print_exc()
            break # Stop training on error

        # --- Logging ---
        current_time = time.time()
        steps_since_log += 1
        if current_time - last_log_time >= args.log_interval_sec or step == args.total_steps - 1:
            elapsed_time = current_time - start_time
            steps_per_sec = steps_since_log / (current_time - last_log_time) if (current_time - last_log_time) > 0 else 0
            total_estimated_time = elapsed_time / (step + 1) * args.total_steps if step > 0 else 0
            eta_seconds = total_estimated_time - elapsed_time if total_estimated_time > elapsed_time else 0

            pbar.set_postfix({
                "Gt_log": f"{metrics.get('Gt_log', float('nan')):.2f}",
                "ModLoss": f"{metrics.get('module_loss_avg', float('nan')):.2f}", # Total loss
                "ModGrad": f"{metrics.get('module_grad_norm_avg', float('nan')):.2f}",
                "CLS Norm": f"{metrics.get('cls_norm', float('nan')):.1f}",
                "Gamma": f"{metrics.get('gamma_t', float('nan')):.3f}",
                "LE_EMA": f"{metrics.get('lambda_max_EMA', float('nan')):.1f}",
                "Steps/s": f"{steps_per_sec:.1f}",
                "ETA": f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            })
            last_log_time = current_time
            steps_since_log = 0

        # --- OOD Evaluation ---
        if args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
             ood_acc, ood_loss = evaluate_ood(system, env, args.eval_steps)
             logger.log_metrics({"OOD_Accuracy": ood_acc, "OOD_Loss": ood_loss}, step)
             tqdm.write(f"Step {step}: OOD Accuracy: {ood_acc:.4f}, OOD Loss: {ood_loss:.4f}")


    # --- Save final state and plot ---
    logger.save_log()
    final_checkpoint_path = os.path.join(args.log_dir, "final_model_state.pth")
    torch.save(system.state_dict(), final_checkpoint_path)
    print(f"Final model state saved to {final_checkpoint_path}")
    try:
        plot_metrics(logger.log_file)
        print(f"Metrics plot saved to {os.path.join(args.log_dir, 'metrics_plot.png')}")
    except Exception as e:
        print(f"Failed to generate metrics plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD Model")
    # Model Params
    parser.add_argument('--cls-dim', type=int, default=4096, help='Dimension of Central Latent Space')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of generic predictive modules')
    parser.add_argument('--module-hidden-dim', type=int, default=128, help='Hidden dimension for predictive modules')
    parser.add_argument('--meta-hidden-dim', type=int, default=64, help='Hidden dimension for meta-model')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Dimension for task input embeddings')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity factor for CLS writes (fraction of active dims)')
    # Training Params
    parser.add_argument('--total-steps', type=int, default=20000, help='Total number of training steps') # Increased steps
    parser.add_argument('--module-lr', type=float, default=5e-5, help='Learning rate for predictive modules') # Keep low
    parser.add_argument('--meta-lr', type=float, default=5e-5, help='Learning rate for meta-model') # Keep low
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for Adam optimizers')
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Coefficient for entropy bonus in module loss')
    parser.add_argument('--action-std', type=float, default=DEFAULT_ACTION_STD, help='Standard deviation for module action sampling') # Use default from core.py
    # --- MODIFICATION: Added prediction loss weight arg ---
    parser.add_argument('--prediction-loss-weight', type=float, default=DEFAULT_PREDICTION_LOSS_WEIGHT, help='Weight for the prediction loss component in modules')
    # --- END MODIFICATION ---
    # Dynamics Params
    parser.add_argument('--gamma-min', type=float, default=0.01, help='Min value for CLS decay rate gamma_t')
    parser.add_argument('--gamma-max', type=float, default=0.2, help='Max value for CLS decay rate gamma_t')
    parser.add_argument('--noise-start', type=float, default=0.05, help='Initial noise standard deviation')
    parser.add_argument('--noise-end', type=float, default=0.001, help='Final noise standard deviation')
    parser.add_argument('--ema-alpha', type=float, default=0.99, help='EMA decay factor for smoothing Gt and lambda_max')
    parser.add_argument('--no-cls-clip', action='store_true', help='Disable CLS norm clipping')
    # Stability Params
    # --- MODIFICATION: Re-enable LE estimation ---
    parser.add_argument('--le-interval', type=int, default=2000, help='Interval for estimating Lyapunov exponent (-1 to disable)')
    # --- END MODIFICATION ---
    parser.add_argument('--stability-target', type=float, default=-0.01, help='Target value for lambda_max_EMA for Meta-Model reward') # Target slightly negative
    # Environment Params
    parser.add_argument('--max-val', type=int, default=100, help='Maximum value for numbers in arithmetic task')
    # Logging/Eval Params
    parser.add_argument('--log-dir', type=str, default='logs_p1_rev7_combined_loss', help='Directory for logs and checkpoints') # Updated dir name
    parser.add_argument('--log-interval-sec', type=int, default=5, help='Log progress every N seconds')
    parser.add_argument('--eval-interval', type=int, default=5000, help='Evaluate OOD performance every N steps (-1 to disable)') # Re-enable OOD eval
    parser.add_argument('--eval-steps', type=int, default=500, help='Number of steps for OOD evaluation')
    # System Params
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    main(args)