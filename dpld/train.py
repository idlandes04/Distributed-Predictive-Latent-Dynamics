import torch
import argparse
import numpy as np
import os

from core import DPLDSystem
from envs import LorenzEnv
from utils import Logger, plot_metrics

def noise_schedule_const(step, std_dev=0.05):
    """Constant noise schedule."""
    return std_dev

def noise_schedule_anneal(step, start_std=0.1, end_std=0.01, anneal_steps=10000):
    """Linearly annealing noise schedule."""
    anneal_frac = min(1.0, step / anneal_steps)
    return start_std - (start_std - end_std) * anneal_frac

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # --- Environment ---
    # For MVP, environment is just used to potentially initialize CLS,
    # but DPLD runs its internal dynamics prediction loop.
    # env = LorenzEnv(device=device)
    # env_dim = env.get_dimension()
    # We could have a module that takes Lorenz state as input, but MVP focuses
    # on CLS predicting its own next state based on internal dynamics.

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
    for step in range(args.total_steps):
        # Run one step of the DPLD internal dynamics
        metrics = dpld_system.step(step)

        # Logging
        if step % args.log_interval == 0:
            logger.log(step, metrics)

        # Optional: Checkpointing
        # if step % args.save_interval == 0 and step > 0:
        #     save_path = os.path.join(logger.log_dir, f"model_step_{step}.pt")
        #     torch.save(dpld_system.state_dict(), save_path)
        #     print(f"Checkpoint saved to {save_path}")

        # Optional: Estimate Lyapunov exponent periodically (can be slow)
        # if step % args.le_interval == 0 and step > 0:
        #     print("Estimating Lyapunov Exponent...")
        #     dynamics_map_fn = dpld_system.get_dynamics_map()
        #     le_estimate = estimate_lyapunov_exponent(dynamics_map_fn, dpld_system.ct, device=device)
        #     print(f"Step {step} Estimated LE: {le_estimate:.4f}")
        #     # Could log this LE estimate as well
        #     logger.log(step, {"lambda_max_estimated_periodic": le_estimate})


    print("Training finished.")
    logger.close()

    # --- Plotting ---
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
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Learning rate for meta-model')
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
    parser.add_argument('--le-interval', type=int, default=1000, help='Steps between estimating LE (can be slow)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for logs and checkpoints')
    parser.add_argument('--run-name', type=str, default=None, help='Specific name for this run')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    # parser.add_argument('--save-interval', type=int, default=5000, help='Steps between saving model checkpoints')


    args = parser.parse_args()
    main(args)