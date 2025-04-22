# dpld/train.py

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os
import time
import math
from collections import deque

# Import revised core components
from core import DPLDSystem, DEFAULT_ACTION_STD, CLS_NORM_CLIP_VAL, GRAD_CLIP_NORM # Use updated constants
from envs import ArithmeticEnv
from utils import Logger, plot_metrics # Logger and plot_metrics assumed unchanged

def noise_schedule_const(step, std_dev=0.05):
    return std_dev

def noise_schedule_anneal(step, start_std=0.1, end_std=0.01, anneal_steps=10000):
    if anneal_steps <= 0: return start_std
    anneal_frac = min(1.0, step / anneal_steps)
    return start_std - (start_std - end_std) * anneal_frac

def evaluate_ood(dpld_system, ood_env, eval_steps=100):
    """Evaluates the system on the OOD environment (Task evaluation disabled for Rev 4)."""
    print("\n--- Starting OOD Evaluation (CLS Dynamics Only) ---")
    dpld_system.eval() # Set system to evaluation mode
    # accuracies = [] # No accuracy calculation
    # task_losses_raw = [] # No task loss calculation
    ood_env.set_range('ood', min_val=ood_env.current_min_val, max_val=ood_env.current_max_val)

    # Reset hidden states for evaluation
    with torch.no_grad():
        if hasattr(dpld_system.task_head, 'last_gru_hidden_state'):
            dpld_system.task_head.last_gru_hidden_state = None
        if hasattr(dpld_system.meta_model, 'last_gru_hidden_state'):
            dpld_system.meta_model.last_gru_hidden_state = None
        # Reset stored previous surprises
        dpld_system.last_log_surprises_sm = None

        for _ in range(eval_steps):
            task_input = ood_env.step()
            a, op_idx, b, true_answer_c = task_input

            # --- Simulate system step without learning ---
            # 1. Predict (updates internal state like GRU hidden)
            all_modules = dpld_system.pred_modules + [dpld_system.task_head]
            for module in all_modules:
                _ = module.predict(dpld_system.ct)

            # Get task prediction for accuracy/loss calculation - SKIPPED
            # task_answer_prediction = dpld_system.task_head.get_task_prediction()

            # Calculate accuracy and raw task loss - SKIPPED

            # 2. Meta-Model Regulation (use EMA state if available, otherwise defaults)
            gt_log_input = dpld_system.gt_log_ema if dpld_system.gt_log_ema is not None else 0.0
            lambda_input = dpld_system.lambda_max_ema if dpld_system.lambda_max_ema is not None else dpld_system.meta_model.stability_target
            gamma_t, mmod_t = dpld_system.meta_model.compute_regulation(gt_log_input, lambda_input)

            # 3. Generate Write Vectors (using internal predictions)
            sum_im = dpld_system._init_cls()
            i_math = dpld_system.math_encoder(a, op_idx, b)
            sum_im += i_math
            for module in all_modules:
                 im = module.generate_write_vector(dpld_system.ct)
                 module.last_log_prob = None
                 module.last_action_dist = None
                 sum_im += im
            sum_im = sum_im.coalesce()

            # 4. CLS Update (use minimal noise during eval?)
            noise_std_dev = dpld_system.noise_std_dev_schedule(eval_steps)
            noise_std_dev = min(noise_std_dev, 0.01) # Use low noise for eval
            dpld_system.ct = dpld_system.cls_update_rule(dpld_system.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

            # 5. Calculate Surprise (needed to update baselines, but not stored for DR)
            current_eval_log_surprises = []
            for module in all_modules:
                 # TaskHead now calculates only CLS surprise
                 sm_log = module.calculate_surprise(dpld_system.ct, true_answer_c=None) # Pass None for true_answer
                 current_eval_log_surprises.append(sm_log.detach())
            # Store these for the *next* eval step's DR calculation if we were doing DR here
            # dpld_system.last_log_surprises_sm = [s.detach().clone() for s in current_eval_log_surprises]

    # avg_accuracy = np.mean(accuracies) if accuracies else 0.0 # Skipped
    # finite_losses = [l for l in task_losses_raw if math.isfinite(l)] # Skipped
    # avg_task_loss_raw = np.mean(finite_losses) if finite_losses else float('nan') # Skipped

    print(f"--- OOD Evaluation Complete (No Task Metrics) ---")
    # print(f"  Avg Accuracy: {avg_accuracy:.4f}") # Skipped
    # print(f"  Avg Raw Task MSE Loss: {avg_task_loss_raw:.4f}\n") # Skipped

    dpld_system.train() # Set system back to training mode
    ood_env.set_range('train') # Switch env back to training range
    # Crucially, reset the stored surprises after eval so training starts fresh
    dpld_system.last_log_surprises_sm = None
    # Return dummy values as task eval is disabled
    return 0.0, float('nan')


def main(args):
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.use_gpu:
        print("Warning: --use-gpu specified but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    else:
        print("Using CPU.")
        device = torch.device("cpu")

    train_env = ArithmeticEnv(min_val=args.train_min, max_val=args.train_max, device=device)
    max_val_overall = args.train_max
    if args.ood_interval > 0:
        max_val_overall = max(args.train_max, args.ood_max)
    if max_val_overall > train_env._max_val_overall:
        train_env._max_val_overall = max_val_overall
        train_env.vocab_size_numbers = max_val_overall + 1

    if args.noise_anneal > 0:
        noise_sched_fn = lambda step: noise_schedule_anneal(step, args.noise_start, args.noise_end, args.noise_anneal)
        print(f"Using annealing noise: start={args.noise_start}, end={args.noise_end}, steps={args.noise_anneal}")
    else:
        noise_sched_fn = lambda step: noise_schedule_const(step, args.noise_start)
        print(f"Using constant noise: std_dev={args.noise_start}")

    dpld_system = DPLDSystem(
        cls_dim=args.cls_dim,
        num_modules=args.num_modules,
        module_hidden_dim=args.module_hidden,
        meta_hidden_dim=args.meta_hidden,
        k_sparse_write=args.k_sparse,
        module_lr=args.module_lr,
        meta_lr=args.meta_lr,
        noise_std_dev_schedule=noise_sched_fn,
        env=train_env,
        embedding_dim=args.embedding_dim,
        entropy_coeff=args.entropy_coeff,
        ema_alpha=args.ema_alpha,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        stability_target=args.stability_target, # Use potentially revised default
        action_std=args.action_std, # Use potentially revised default
        task_loss_weight=args.task_loss_weight, # Use potentially revised default (0.0)
        weight_decay=args.weight_decay, # Use potentially revised default
        meta_input_dim=2, # Gt_log_EMA, lambda_max_EMA
        clip_cls_norm=not args.no_clip_cls_norm,
        device=device
    ).to(device)

    print(f"DPLD System initialized (Phase 1 Revision 4 - Radical Simplification):")
    print(f"  CLS Dim: {args.cls_dim}, Num Generic Modules: {args.num_modules}")
    print(f"  Module Hidden: {args.module_hidden}, Meta Hidden: {args.meta_hidden}")
    print(f"  Embedding Dim: {args.embedding_dim}, Write Sparsity (k): {args.k_sparse}")
    print(f"  Module LR: {args.module_lr}, Meta LR: {args.meta_lr}, Weight Decay: {args.weight_decay}") # Show weight decay
    print(f"  Entropy Coeff: {args.entropy_coeff}, EMA Alpha: {args.ema_alpha}")
    print(f"  Gamma Range: [{args.gamma_min}, {args.gamma_max}], Stability Target: {args.stability_target}") # Show stability target
    print(f"  Noise Schedule: {'Anneal' if args.noise_anneal > 0 else 'Constant'}, Action Std: {args.action_std}") # Show action std
    print(f"  Task Loss Weight (for log-surprise): {args.task_loss_weight}, Clip CLS Norm: {not args.no_clip_cls_norm}") # Show task weight
    print(f"  Using LOG-SURPRISE and Simple DR (Rm = -Sm_log). TASK LEARNING DISABLED.")
    print(f"  Gradient Clipping Norm: {GRAD_CLIP_NORM}") # Show grad clip norm


    logger = Logger(log_dir=args.log_dir, run_name=args.run_name)
    # Define header keys including new/revised metrics (Task metrics removed)
    initial_header_keys = [
        "step",
        # Log-Surprise Based (Only CLS)
        "Gt_log", "Gt_log_EMA", "Sm_log_avg", "Sm_log_std",
        "Sm_log_cls_avg", #"Sm_log_task_avg", "TaskHead_Sm_log_task", # Removed
        # Raw Surprise Based (Only CLS)
        "Sm_raw_cls_avg", #"Sm_raw_task_avg", "TaskHead_Sm_raw_task", # Removed
        # Task Performance - REMOVED
        # "TaskAccuracy", "TaskAcc_Recent",
        # Reward Based (Simple DR: -Sm_log)
        "Rm_log_raw_avg", "Rm_log_raw_std", "Rm_norm_avg", "Rm_norm_std",
        # Stability & Dynamics
        "lambda_max_est", "lambda_max_EMA", "gamma_t", "noise_std",
        # Learning Performance
        "module_loss_avg", "meta_loss",
        "module_entropy_avg", "module_grad_norm_avg", "meta_grad_norm",
        # CLS State
        "cls_norm", "cls_density"
    ]
    if args.ood_interval > 0:
         # OOD metrics are disabled for now
         logger.header_keys = initial_header_keys # + ["OOD_Accuracy", "OOD_TaskLoss_Raw"]
    else:
         logger.header_keys = initial_header_keys

    print("Starting training...")
    start_time = time.time()
    # recent_accuracy = deque(maxlen=args.log_interval) # Disabled
    ood_env = None

    for step in range(args.total_steps):
        dpld_system.train() # Ensure model is in training mode
        estimate_le_this_step = (args.le_interval > 0 and step % args.le_interval == 0 and step >= args.le_warmup)
        task_input = train_env.step() # Still need input for encoder

        try:
            # Pass None for true_answer_c as task learning is disabled
            metrics = dpld_system.step(step, task_input=task_input, estimate_le=estimate_le_this_step, true_answer_c=None)
            if metrics is None:
                 print(f"Warning: dpld_system.step returned None at step {step}. Skipping log.")
                 continue

            # if 'TaskAccuracy' in metrics and math.isfinite(metrics['TaskAccuracy']): # Disabled
            #      recent_accuracy.append(metrics['TaskAccuracy'])
            # metrics['TaskAcc_Recent'] = np.mean(recent_accuracy) if recent_accuracy else 0.0 # Disabled

        except Exception as e:
             print(f"\nERROR during DPLD step {step}: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             save_dir = os.path.join(logger.log_dir, "checkpoints")
             os.makedirs(save_dir, exist_ok=True)
             save_path = os.path.join(save_dir, f"model_step_{step}_ABORT.pt")
             try: torch.save(dpld_system.state_dict(), save_path); print(f"Saved state to {save_path}")
             except Exception as save_e: print(f"Error saving state: {save_e}")
             break # Exit training loop

        ood_metrics = {}
        if args.ood_interval > 0 and step % args.ood_interval == 0 and step >= args.ood_warmup:
            if ood_env is None:
                 ood_env = ArithmeticEnv(min_val=args.ood_min, max_val=args.ood_max, device=device)
                 if ood_env._max_val_overall > train_env._max_val_overall:
                     train_env._max_val_overall = ood_env._max_val_overall
                     train_env.vocab_size_numbers = ood_env.vocab_size_numbers
                     print(f"Warning: OOD max value {ood_env._max_val_overall} > Train max value {train_env._max_val_overall}. Ensure embeddings cover full range.")

            # OOD evaluation is now just running dynamics, no task metrics returned
            _, _ = evaluate_ood(dpld_system, ood_env, eval_steps=args.ood_steps)
            # ood_metrics = {"OOD_Accuracy": ood_acc, "OOD_TaskLoss_Raw": ood_loss_raw} # Disabled

        # Logging
        if step % args.log_interval == 0 or step == args.total_steps - 1:
            combined_metrics = {**metrics, **ood_metrics} # Combine metrics (ood_metrics will be empty)
            logger.log(step, combined_metrics) # Log combined dict

            if step > 0 and args.log_interval > 0:
                 current_time = time.time()
                 elapsed_since_last_log = current_time - logger.last_log_time
                 steps_per_sec = args.log_interval / elapsed_since_last_log if elapsed_since_last_log > 1e-3 else 0
                 logger.last_log_time = current_time
                 remaining_steps = args.total_steps - (step + 1)
                 if steps_per_sec > 1e-3:
                     eta_seconds = remaining_steps / steps_per_sec
                     eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                 else:
                     eta_str = ">1 day"
                 print(f"    ETA: {eta_str} ({steps_per_sec:.2f} steps/sec)")

        # Checkpointing
        if args.save_interval > 0 and step % args.save_interval == 0 and step > 0:
            save_dir = os.path.join(logger.log_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_step_{step}.pt")
            try: torch.save(dpld_system.state_dict(), save_path); print(f"    Checkpoint saved to {save_path}")
            except Exception as e: print(f"    Error saving checkpoint: {e}")

        # Instability Check (Removed Task Loss checks)
        if metrics is not None and (
            ('Gt_log' in metrics and metrics['Gt_log'] is not None and not math.isfinite(metrics['Gt_log'])) or
            ('cls_norm' in metrics and metrics['cls_norm'] is not None and (not math.isfinite(metrics['cls_norm']) or metrics['cls_norm'] > CLS_NORM_CLIP_VAL * 1.5)) or
            ('module_loss_avg' in metrics and metrics['module_loss_avg'] is not None and abs(metrics['module_loss_avg']) > 1e4) or
            ('module_grad_norm_avg' in metrics and metrics['module_grad_norm_avg'] is not None and (not math.isfinite(metrics['module_grad_norm_avg']) or metrics['module_grad_norm_avg'] > 100.0)) # Reduced threshold for raw grad norm avg
        ):
             print(f"\nERROR: Potential instability detected at step {step}.")
             print(f"  Gt_log: {metrics.get('Gt_log')}, cls_norm: {metrics.get('cls_norm')}, module_loss_avg: {metrics.get('module_loss_avg')}, module_grad_norm_avg: {metrics.get('module_grad_norm_avg')}")
             print("Aborting training.")
             save_dir = os.path.join(logger.log_dir, "checkpoints")
             os.makedirs(save_dir, exist_ok=True)
             save_path = os.path.join(save_dir, f"model_step_{step}_INSTABILITY.pt")
             try: torch.save(dpld_system.state_dict(), save_path); print(f"Saved state to {save_path}")
             except Exception as save_e: print(f"Error saving state: {save_e}")
             break # Exit training loop

    print("Training finished.")
    total_time = time.time() - start_time
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.close()

    print("Generating plot...")
    try:
        # Adjust metrics to plot based on removed task metrics
        metrics_to_plot = [
            "Gt_log", "Gt_log_EMA", "Sm_log_avg", "Sm_log_std", "Sm_log_cls_avg",
            "Sm_raw_cls_avg",
            "Rm_log_raw_avg", "Rm_log_raw_std", "Rm_norm_avg", "Rm_norm_std",
            "lambda_max_est", "lambda_max_EMA", "gamma_t", "noise_std",
            "module_loss_avg", "meta_loss",
            "module_entropy_avg", "module_grad_norm_avg", "meta_grad_norm",
            "cls_norm", "cls_density"
        ]
        plot_metrics(logger.csv_path, metrics_to_plot=metrics_to_plot)
    except Exception as e:
        print(f"Error generating plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD Phase 1 Revision 4 (Radical Simplification)")

    # DPLD Architecture
    parser.add_argument('--cls-dim', type=int, default=512, help='Dimension of Central Latent Space (D)')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of *generic* predictive modules (M_generic)')
    parser.add_argument('--module-hidden', type=int, default=128, help='Hidden dimension for module MLP/GRU')
    parser.add_argument('--meta-hidden', type=int, default=64, help='Hidden dimension for meta-model GRU')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dimension for arithmetic encoder')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity fraction for module writes (k)')

    # Learning Parameters (Revised Defaults for Rev 4)
    parser.add_argument('--module-lr', type=float, default=1e-4, help='Learning rate for modules')
    parser.add_argument('--meta-lr', type=float, default=5e-5, help='Learning rate for meta-model')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for Adam optimizers') # Increased default
    parser.add_argument('--entropy-coeff', type=float, default=0.01, help='Coefficient for entropy bonus in module loss') # Reduced default
    parser.add_argument('--ema-alpha', type=float, default=0.99, help='Alpha for EMA smoothing of Meta-Model inputs')
    parser.add_argument('--total-steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--task-loss-weight', type=float, default=0.0, help='Weight for task log-surprise (DISABLED FOR REV 4)') # Disabled

    # Dynamics Parameters (Revised Defaults for Rev 4)
    parser.add_argument('--gamma-min', type=float, default=0.05, help='Minimum global decay rate')
    parser.add_argument('--gamma-max', type=float, default=0.3, help='Maximum global decay rate')
    parser.add_argument('--stability-target', type=float, default=-100.0, help='Target for lambda_max (High neg to ignore LE)') # Revised default
    parser.add_argument('--noise-start', type=float, default=0.05, help='Initial/constant noise std dev')
    parser.add_argument('--noise-end', type=float, default=0.01, help='Final noise std dev for annealing')
    parser.add_argument('--noise-anneal', type=int, default=50000, help='Steps to anneal noise over (0 for constant)')
    parser.add_argument('--action-std', type=float, default=0.1, help='Std dev for sampling module write actions') # Reduced default
    parser.add_argument('--no-clip-cls-norm', action='store_true', help='Disable CLS norm clipping')

    # Task Parameters (OOD eval is now dynamics-only)
    parser.add_argument('--train-min', type=int, default=1, help='Min number for training')
    parser.add_argument('--train-max', type=int, default=100, help='Max number for training')
    parser.add_argument('--ood-min', type=int, default=101, help='Min number for OOD testing')
    parser.add_argument('--ood-max', type=int, default=200, help='Max number for OOD testing')
    parser.add_argument('--ood-interval', type=int, default=10000, help='Steps between OOD evaluations (0 to disable)')
    parser.add_argument('--ood-steps', type=int, default=500, help='Number of steps for each OOD evaluation')
    parser.add_argument('--ood-warmup', type=int, default=10000, help='Steps before starting OOD evaluation')


    # Logging & Setup
    parser.add_argument('--log-interval', type=int, default=200, help='Steps between logging metrics')
    parser.add_argument('--le-interval', type=int, default=5000, help='Steps between estimating LE (0 to disable)')
    parser.add_argument('--le-warmup', type=int, default=5000, help='Steps before starting LE estimation')
    parser.add_argument('--save-interval', type=int, default=0, help='Steps between saving model checkpoints (0 to disable)')
    parser.add_argument('--log-dir', type=str, default='logs_p1_rev4', help='Directory for logs and checkpoints') # New default dir
    parser.add_argument('--run-name', type=str, default=None, help='Specific name for this run')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    main(args)