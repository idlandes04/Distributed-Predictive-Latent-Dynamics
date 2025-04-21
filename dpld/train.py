import torch
import argparse
import numpy as np
import os
import time
import math # Import math for checking finite values

# Import necessary components from core.py
from core import DPLDSystem, DEFAULT_ACTION_STD # <<< IMPORT ADDED HERE
from utils import Logger, plot_metrics

def noise_schedule_const(step, std_dev=0.05):
    return std_dev

def noise_schedule_anneal(step, start_std=0.1, end_std=0.01, anneal_steps=10000):
    anneal_frac = min(1.0, step / anneal_steps)
    return start_std - (start_std - end_std) * anneal_frac

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
        meta_lr=args.meta_lr,
        noise_std_dev_schedule=noise_sched_fn,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        stability_target=args.stability_target,
        action_std=args.action_std,
        clip_cls_norm=not args.no_clip_cls_norm,
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
    print(f"  Action Noise Std: {args.action_std}")
    print(f"  Clip CLS Norm: {not args.no_clip_cls_norm}")


    logger = Logger(log_dir=args.log_dir, run_name=args.run_name)

    print("Starting training...")
    start_time = time.time()
    for step in range(args.total_steps):
        estimate_le_this_step = (args.le_interval > 0 and step % args.le_interval == 0)

        try:
            metrics = dpld_system.step(step, estimate_le=estimate_le_this_step)
        except Exception as e:
             print(f"\nERROR during DPLD step {step}: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             break # Stop training loop on error


        if step % args.log_interval == 0 or step == args.total_steps - 1:
            logger.log(step, metrics)
            if step > 0 and args.log_interval > 0 : # Avoid division by zero and excessive printing
                 elapsed_time = time.time() - start_time
                 steps_per_sec = (step + 1) / elapsed_time # Use step+1 for current avg
                 remaining_steps = args.total_steps - (step + 1)
                 # Handle potential division by zero if steps_per_sec is 0
                 eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else float('inf')
                 if math.isfinite(eta_seconds):
                     eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                     print(f"    ETA: {eta_str} ({steps_per_sec:.2f} steps/sec)")
                 else:
                     print(f"    ETA: Inf ({steps_per_sec:.2f} steps/sec)")


        if args.save_interval > 0 and step % args.save_interval == 0 and step > 0:
            save_dir = os.path.join(logger.log_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_step_{step}.pt")
            try:
                torch.save(dpld_system.state_dict(), save_path)
                print(f"    Checkpoint saved to {save_path}")
            except Exception as e:
                print(f"    Error saving checkpoint: {e}")

        # --- Check for NaN in critical metrics ---
        if metrics is not None and (
            ('Gt' in metrics and metrics['Gt'] is not None and not math.isfinite(metrics['Gt'])) or
            ('cls_norm' in metrics and metrics['cls_norm'] is not None and not math.isfinite(metrics['cls_norm']))
        ):
             print(f"\nERROR: Non-finite metric detected at step {step}. Gt: {metrics.get('Gt')}, cls_norm: {metrics.get('cls_norm')}")
             print("Aborting training due to instability.")
             break


    print("Training finished.")
    total_time = time.time() - start_time
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.close()

    print("Generating plot...")
    # Add try-except block for plotting as well
    try:
        plot_metrics(logger.csv_path)
    except Exception as e:
        print(f"Error generating plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPLD MVP")

    # DPLD Architecture
    parser.add_argument('--cls-dim', type=int, default=512, help='Dimension of Central Latent Space (D)')
    parser.add_argument('--num-modules', type=int, default=3, help='Number of predictive modules (M)')
    parser.add_argument('--module-hidden', type=int, default=128, help='Hidden dimension for module MLP')
    parser.add_argument('--meta-hidden', type=int, default=64, help='Hidden dimension for meta-model')
    parser.add_argument('--k-sparse', type=float, default=0.05, help='Sparsity fraction for module writes (k)')

    # Learning Parameters
    parser.add_argument('--module-lr', type=float, default=5e-5, help='Learning rate for modules (reduced default)')
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Learning rate for meta-model')
    parser.add_argument('--total-steps', type=int, default=50000, help='Total training steps')

    # Dynamics Parameters
    parser.add_argument('--gamma-min', type=float, default=0.05, help='Minimum global decay rate (increased default)')
    parser.add_argument('--gamma-max', type=float, default=0.3, help='Maximum global decay rate (increased default)')
    parser.add_argument('--stability-target', type=float, default=0.05, help='Target for lambda_max (lambda_thr)')
    parser.add_argument('--noise-start', type=float, default=0.05, help='Initial/constant noise std dev (reduced default)')
    parser.add_argument('--noise-end', type=float, default=0.01, help='Final noise std dev for annealing')
    parser.add_argument('--noise-anneal', type=int, default=10000, help='Steps to anneal noise over (0 for constant)')
    # Use the imported constant for the default value
    parser.add_argument('--action-std', type=float, default=DEFAULT_ACTION_STD, help='Std dev for sampling module write actions')
    parser.add_argument('--no-clip-cls-norm', action='store_true', help='Disable CLS norm clipping')


    # Logging & Setup
    parser.add_argument('--log-interval', type=int, default=100, help='Steps between logging metrics')
    parser.add_argument('--le-interval', type=int, default=1000, help='Steps between estimating LE (0 to disable)')
    parser.add_argument('--save-interval', type=int, default=5000, help='Steps between saving model checkpoints (0 to disable)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for logs and checkpoints')
    parser.add_argument('--run-name', type=str, default=None, help='Specific name for this run')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')


    args = parser.parse_args()
    main(args)