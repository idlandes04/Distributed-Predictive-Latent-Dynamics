import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical # Or others if needed

from utils import estimate_lyapunov_exponent, sparsify_vector # Add sparse utils if needed

# --- Constants ---
EPSILON = 1e-8 # For numerical stability (e.g., log probabilities)

# --- Predictive Module ---
class PredictiveModule(nn.Module):
    """
    Implements a DPLD module: Read, Predict, Write (via sparse, gated contribution).
    Learns via difference reward based on global surprise.
    """
    def __init__(self, cls_dim, module_hidden_dim, k_sparse_write, learning_rate,
                 surprise_scale_factor=1.0, surprise_baseline_ema=0.99, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.k_sparse_write = k_sparse_write # Sparsity fraction k (Part II, Alg 1)
        self.device = device
        self.surprise_scale_factor = surprise_scale_factor # βα in Part II, Alg 1, Eq 4
        self.surprise_baseline_ema = surprise_baseline_ema # For Sm baseline in Alg 1

        # Internal Predictive Model (fm in Part II, Sec 4.1) - Simple MLP
        self.fm = nn.Sequential(
            nn.Linear(cls_dim, module_hidden_dim),
            nn.ReLU(),
            nn.Linear(module_hidden_dim, module_hidden_dim),
            nn.ReLU(),
             # Output layer predicts the *entire* next CLS state for MVP simplicity
            nn.Linear(module_hidden_dim, cls_dim)
        ).to(device)
        # Parameters θm are implicitly self.fm.parameters()

        # Gating Query Vector (qm in Part II, Alg 1)
        self.qm = nn.Parameter(torch.randn(cls_dim, device=device) * 0.1)

        # Store running average of surprise Sm (Sm_bar in Alg 1 logic)
        self.register_buffer('sm_baseline', torch.tensor(1.0, device=device)) # Initialize baseline reasonably

        # Store last prediction, surprise, and write vector for learning
        self.last_prediction_ct_plus_1 = None # ĉm,t+1
        self.last_surprise_sm = None          # Sm
        self.last_write_vector_im = None      # Im
        self.last_log_prob = None             # Log prob of action (write vector) for REINFORCE

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def predict(self, ct):
        """Predicts the next CLS state based on the current state ct."""
        # ct is expected to be a sparse tensor, convert to dense for MLP
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        self.last_prediction_ct_plus_1 = self.fm(ct_dense) # ĉm,t+1
        return self.last_prediction_ct_plus_1

    def calculate_surprise(self, actual_ct_plus_1):
        """Calculates local surprise Sm based on prediction and actual next state."""
        # Ensure actual_ct_plus_1 is dense for loss calculation
        actual_ct_plus_1_dense = actual_ct_plus_1.to_dense() if actual_ct_plus_1.is_sparse else actual_ct_plus_1

        if self.last_prediction_ct_plus_1 is None:
             raise RuntimeError("Must call predict() before calculate_surprise()")

        # Using Mean Squared Error as the distance metric (Eq 2, Part II)
        # Note: Paper uses Distance(ĉ, Proj(c)). Here Proj is identity for MVP.
        surprise = F.mse_loss(self.last_prediction_ct_plus_1, actual_ct_plus_1_dense, reduction='mean')

        self.last_surprise_sm = surprise

        # Update baseline Sm (Sm_bar) using EMA
        self.sm_baseline = self.surprise_baseline_ema * self.sm_baseline + \
                           (1 - self.surprise_baseline_ema) * surprise.detach()

        return self.last_surprise_sm

    def generate_write_vector(self, ct):
        """Generates the sparse, weighted, gated write vector Im (Alg 1, Part II)."""
        if self.last_prediction_ct_plus_1 is None or self.last_surprise_sm is None:
             raise RuntimeError("Must call predict() and calculate_surprise() before generate_write_vector()")

        # Alg 1, Step 1: Project module output (vm = ĉm,t+1 in MVP)
        # Wm is identity here, so vm = self.last_prediction_ct_plus_1
        vm = self.last_prediction_ct_plus_1 # Raw output vector

        # Alg 1, Step 2: Compute raw gating score (sm = qm^T * ct)
        ct_dense = ct.to_dense() if ct.is_sparse else ct
        raw_gate_score = torch.dot(self.qm, ct_dense) # sm (scalar score for simplicity)
        # Alternative: element-wise gating gm = sigmoid(qm * ct / tau_g)? Paper implies vector gm.
        # Let's try element-wise gating for more expressivity.
        # raw_gate_score_vec = self.qm * ct_dense # Element-wise product

        # Alg 1, Step 3: Compute element-wise gate activation (gm = sigmoid(sm/τg))
        # Using vector version: gm = sigmoid(qm * ct / tau_g)
        tau_g = 1.0 # Gating temperature, hyperparameter
        gate_activation_gm = torch.sigmoid(self.qm * ct_dense / tau_g) # gm (vector [0,1]^D)

        # Alg 1, Step 4: Modulate influence by surprise (αm = α_base + α_scale * tanh(βα(Sm - Sm_bar)))
        alpha_base = 1.0 # Base influence
        alpha_scale = 1.0 # Scaling factor for surprise modulation
        # surprise_scale_factor is βα
        surprise_diff = self.last_surprise_sm - self.sm_baseline
        influence_scalar_am = alpha_base + alpha_scale * torch.tanh(self.surprise_scale_factor * surprise_diff)
        influence_scalar_am = torch.clamp(influence_scalar_am, min=0.1) # Ensure non-negative influence

        # Alg 1, Step 5: Apply gating and scaling (Im_dense = αm * (gm ⊙ vm))
        # Using Hadamard product (element-wise)
        intermediate_write_vector = influence_scalar_am * (gate_activation_gm * vm) # Dense intermediate Im

        # --- Stochasticity for REINFORCE ---
        # Option 1: Add noise to vm before gating/scaling (simple)
        # Option 2: Output parameters of a distribution from fm, sample vm
        # Option 3: Make sparse index selection stochastic (complex)
        # Let's use Option 1 for MVP: treat the deterministic intermediate_write_vector
        # as the mean of a Normal distribution, sample from it, then sparsify.
        # This action allows gradient flow via REINFORCE.
        action_mean = intermediate_write_vector
        action_std = 0.1 # Fixed std deviation for simplicity, could be learned/scheduled
        dist = Normal(action_mean, action_std)
        # Sample the action (dense vector before sparsification)
        dense_write_vector_sampled = dist.sample()
        self.last_log_prob = dist.log_prob(dense_write_vector_sampled).sum() # Sum log prob over dimensions

        # Alg 1, Step 6 & 7: Sparsify the contribution vector
        self.last_write_vector_im = sparsify_vector(dense_write_vector_sampled, self.k_sparse_write)

        # Convert to sparse tensor format for efficiency
        sparse_indices = torch.where(self.last_write_vector_im != 0)[0].unsqueeze(0)
        sparse_values = self.last_write_vector_im[sparse_indices.squeeze(0)]

        if sparse_indices.numel() > 0:
             im_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, (self.cls_dim,), device=self.device)
        else:
             # Handle case where vector becomes all zero after sparsification
             im_sparse = torch.sparse_coo_tensor((1, 0), [], (self.cls_dim,), device=self.device)


        return im_sparse

    def learn(self, difference_reward_rm):
        """Updates module parameters using the difference reward."""
        if self.last_log_prob is None:
             print("Warning: learn() called before generate_write_vector() produced log_prob. Skipping update.")
             return torch.tensor(0.0) # Return zero loss

        # REINFORCE update rule: loss = - R * log_prob (gradient ascent maximizes R * log_prob)
        # We minimize the negative, hence - R * log_prob
        # R is the difference reward Rm (Prop 4.2, Eq 4, Part II)
        loss = -difference_reward_rm * self.last_log_prob

        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Clear stored values
        self.last_prediction_ct_plus_1 = None
        self.last_surprise_sm = None
        self.last_write_vector_im = None
        self.last_log_prob = None

        return loss.item()


# --- Meta-Model ---
class MetaModel(nn.Module):
    """
    Implements a simplified DPLD Meta-Model for stability regulation.
    Monitors CLS dynamics (lambda_max) and adjusts global decay (gamma_t).
    """
    def __init__(self, cls_dim, meta_hidden_dim, learning_rate,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.device = device
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.stability_target = stability_target # lambda_thr in Thm 7.1

        # Model to process CLS history and predict stability / control params
        # Input: Concatenated recent CLS states (e.g., ct, ct-1) -> 2 * cls_dim
        # For MVP, let's just use the current state ct to decide gamma
        # A more complex version would use an RNN or look at lambda_max history
        self.controller = nn.Sequential(
             # Input size needs adjustment if using history
            nn.Linear(cls_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, 1) # Output: single value controlling gamma
        ).to(device)
        # Parameters θMM are implicitly self.controller.parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.last_lambda_max = None


    def estimate_stability(self, cls_dynamics_map_fn, current_ct):
        """Estimates lambda_max using external utility function."""
        # cls_dynamics_map_fn: function(ct) -> ct+1
        # Needs the *full* DPLD one-step dynamics function
        self.last_lambda_max = estimate_lyapunov_exponent(
            cls_dynamics_map_fn,
            current_ct.detach().to_dense(), # Needs dense state for JVP
            n_vectors=5, # Hyperparameter
            steps=20,    # Hyperparameter (lower for speed in training loop)
            device=self.device
        )
        return self.last_lambda_max

    def compute_regulation(self, current_ct):
        """Computes adaptive decay gamma_t and modulatory vector mmod_t."""
        ct_dense = current_ct.to_dense() if current_ct.is_sparse else current_ct

        # --- Adaptive Decay γt ---
        # Simple strategy: If lambda_max > target, increase decay, else decrease.
        # Use the controller network to map state to a gamma value.
        # Alternative: Directly use estimated lambda_max
        if self.last_lambda_max is not None:
             # Sigmoid mapping from lambda_max to gamma range
             # Higher lambda_max -> higher gamma (closer to gamma_max)
             gamma_signal = torch.sigmoid(torch.tensor(self.last_lambda_max - self.stability_target, device=self.device)) # Shifted sigmoid
             gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * gamma_signal
        else:
             # Default gamma if stability not estimated yet
             gamma_t = (self.gamma_min + self.gamma_max) / 2.0

        # --- Modulatory Vector mmod_t ---
        # Set to zero for MVP
        mmod_t = torch.sparse_coo_tensor((1, 0), [], (self.cls_dim,), device=self.device)

        return gamma_t.item(), mmod_t # Return gamma as scalar, mmod as sparse tensor


    def learn(self):
        """Updates Meta-Model parameters based on stability objective."""
        if self.last_lambda_max is None:
            return 0.0 # Cannot learn without stability estimate

        # Objective LMM: Minimize instability penalty (Eq 5, Part II, simplified)
        # Penalize lambda_max exceeding the target threshold
        instability_penalty = F.relu(torch.tensor(self.last_lambda_max - self.stability_target, device=self.device))
        loss = instability_penalty # Simple ReLU penalty

        # Note: For the controller network to learn, the loss needs to depend
        # on its parameters. Here, gamma_t calculation depends on lambda_max,
        # which depends on the system dynamics influenced by gamma_t from the
        # *previous* step (or controller output if it directly output gamma).
        # This credit assignment is tricky.
        # MVP Simplification: Don't train the controller network yet.
        # Just use the rule-based gamma adaptation based on lambda_max.
        # We can add learning later if needed.
        # If we were training the controller:
        # self.optimizer.zero_grad()
        # loss.backward() # Requires gradient path from lambda_max back to controller params - complex!
        # self.optimizer.step()

        self.last_lambda_max = None # Reset after use
        return loss.item()


# --- DPLD System ---
class DPLDSystem(nn.Module):
    """
    The main DPLD system coordinating CLS, Modules, and Meta-Model.
    """
    def __init__(self, cls_dim, num_modules, module_hidden_dim, meta_hidden_dim,
                 k_sparse_write, module_lr, meta_lr, noise_std_dev_schedule,
                 gamma_min=0.01, gamma_max=0.2, stability_target=0.1, device='cpu'):
        super().__init__()
        self.cls_dim = cls_dim
        self.num_modules = num_modules
        self.k_sparse_write = k_sparse_write
        self.noise_std_dev_schedule = noise_std_dev_schedule # Function step -> std_dev
        self.device = device

        # Initialize CLS state (sparse tensor)
        self.ct = self._init_cls()

        # Initialize Modules
        self.modules = nn.ModuleList([
            PredictiveModule(cls_dim, module_hidden_dim, k_sparse_write, module_lr, device=device)
            for _ in range(num_modules)
        ])

        # Initialize Meta-Model
        self.meta_model = MetaModel(cls_dim, meta_hidden_dim, meta_lr,
                                    gamma_min, gamma_max, stability_target, device=device)

        # Buffer for difference reward calculation
        self.last_gt = None

    def _init_cls(self):
        # Start with a zero or small random sparse vector
        # return torch.sparse_coo_tensor((self.cls_dim,), device=self.device)
        initial_dense = torch.randn(self.cls_dim, device=self.device) * 0.1
        sparse_ct = sparsify_vector(initial_dense, 0.1).to_sparse_coo() # Start sparse
        return sparse_ct


    def cls_update_rule(self, ct, sum_im, mmodt, gamma_t, noise_std_dev):
        """Implements the CLS update equation (Eq 1, Part II)."""
        # Decay term: (1 - gamma_t) * ct
        decayed_ct = (1.0 - gamma_t) * ct

        # Noise term: epsilon_t ~ N(0, sigma_t^2 * I)
        # Generate sparse noise? Or add dense noise then potentially sparsify?
        # Adding dense noise is simpler.
        noise_et = torch.randn(self.cls_dim, device=self.device) * noise_std_dev
        noise_et_sparse = noise_et.to_sparse_coo() # Convert to sparse if needed

        # Combine terms: ct+1 = (1-gamma)ct + Sum(Im) + mmod_t + eps_t
        # Sparse additions
        ct_plus_1 = decayed_ct + sum_im + mmodt + noise_et_sparse

        # Optional: Explicit normalization (Lemma 3.1 discussion)
        # norm = torch.linalg.norm(ct_plus_1.to_dense()) # Requires densification
        # max_norm = 10.0
        # if norm > max_norm:
        #     ct_plus_1 = (ct_plus_1 / norm) * max_norm

        # Ensure result is sparse (additions might densify if indices overlap heavily)
        # Coalesce sums sparse tensors
        ct_plus_1 = ct_plus_1.coalesce()

        # Optional: Re-sparsify if density increases too much
        # current_density = ct_plus_1.values().numel() / self.cls_dim
        # target_density = self.k_sparse_write * self.num_modules # Rough target
        # if current_density > target_density * 1.5:
        #     ct_plus_1 = sparsify_vector(ct_plus_1.to_dense(), target_density).to_sparse_coo()


        return ct_plus_1


    def get_dynamics_map(self):
         """Returns a function representing the one-step CLS dynamics for Lyapunov estimation."""
         # This function needs access to the current state of modules (for Im) and meta-model (for gamma, mmod)
         # It's tricky because these change during training.
         # For estimation, we might need to freeze parameters temporarily or use current ones.

         def dynamics_map(state_t):
             # state_t is assumed dense for JVP calculation
             state_t_sparse = state_t.to_sparse_coo()

             # 1. Get module write vectors Im based on state_t
             sum_im = self._init_cls() # Zero sparse tensor
             with torch.no_grad(): # Don't track gradients through this estimation path
                 for module in self.modules:
                     # Need predict, calc_surprise (dummy here?), generate_write
                     # This is problematic as surprise depends on ct+1.
                     # Approximation: Use current baseline surprise for influence alpha_m?
                     pred = module.predict(state_t_sparse) # Use state_t
                     # Use baseline surprise for alpha_m calculation in generate_write
                     module.last_prediction_ct_plus_1 = pred
                     module.last_surprise_sm = module.sm_baseline # Use baseline
                     im = module.generate_write_vector(state_t_sparse)
                     sum_im += im
                     # Reset module state after use
                     module.last_prediction_ct_plus_1 = None
                     module.last_surprise_sm = None
                     module.last_write_vector_im = None
                     module.last_log_prob = None


                 # 2. Get meta-model regulation based on state_t
                 # Use current lambda_max estimate or a fixed one? Use fixed for map consistency.
                 # gamma_t, mmodt = self.meta_model.compute_regulation(state_t_sparse) # Uses internal lambda_max
                 # Use fixed gamma for map definition:
                 gamma_t = (self.meta_model.gamma_min + self.meta_model.gamma_max) / 2.0
                 mmodt = torch.sparse_coo_tensor((1, 0), [], (self.cls_dim,), device=self.device)


                 # 3. Apply CLS update rule
                 noise_std_dev = 0.0 # No noise for deterministic map estimation
                 next_state = self.cls_update_rule(state_t_sparse, sum_im.coalesce(), mmodt, gamma_t, noise_std_dev)

             return next_state.to_dense() # Return dense for JVP

         return dynamics_map


    def step(self, current_step_num):
        """Performs one full step of the DPLD system interaction."""

        # --- 1. Module Predictions ---
        predictions = []
        for module in self.modules:
            predictions.append(module.predict(self.ct))

        # --- 2. Meta-Model Regulation ---
        # Estimate stability based on current state and dynamics
        # Need the dynamics map function
        # dynamics_map_fn = self.get_dynamics_map() # This might be slow if called every step
        # lambda_max = self.meta_model.estimate_stability(dynamics_map_fn, self.ct)
        # TEMP: Disable Lyapunov estimation during step for speed in MVP training loop
        lambda_max = 0.0 # Placeholder
        self.meta_model.last_lambda_max = lambda_max # Store for logging/gamma calc

        # Compute gamma_t and mmod_t
        gamma_t, mmod_t = self.meta_model.compute_regulation(self.ct)

        # --- 3. Module Write Vectors ---
        # Need surprise first, which depends on ct+1. Chicken and egg.
        # Solution: Calculate surprise *after* ct+1 is computed.
        # But generate_write_vector needs surprise for alpha_m.
        # Approximation: Use surprise from *previous* step (or baseline) for alpha_m.
        # Let's stick to the logic in PredictiveModule which uses current sm_baseline.

        write_vectors_im = []
        sum_im = self._init_cls() # Zero sparse tensor
        for i, module in enumerate(self.modules):
             # predict() was already called. Need dummy surprise calculation to proceed.
             # The real surprise calculation happens after ct+1 is known.
             module.last_prediction_ct_plus_1 = predictions[i]
             module.last_surprise_sm = module.sm_baseline # Use baseline for alpha_m
             im = module.generate_write_vector(self.ct)
             write_vectors_im.append(im)
             sum_im += im
        sum_im = sum_im.coalesce()

        # --- 4. CLS Update ---
        noise_std_dev = self.noise_std_dev_schedule(current_step_num)
        ct_plus_1 = self.cls_update_rule(self.ct, sum_im, mmod_t, gamma_t, noise_std_dev)

        # --- 5. Calculate Actual Surprises & Global Surprise ---
        surprises_sm = []
        global_surprise_gt = 0.0
        for i, module in enumerate(self.modules):
             # Now calculate the actual surprise using ct_plus_1
             sm = module.calculate_surprise(ct_plus_1) # Updates module.last_surprise_sm correctly now
             surprises_sm.append(sm)
             global_surprise_gt += sm
        global_surprise_gt /= self.num_modules

        # --- 6. Calculate Difference Rewards ---
        # Requires counterfactual Gt^{-m} (Def 4.1, Part II)
        difference_rewards_rm = []
        if self.last_gt is not None: # Need previous Gt for comparison? No, need counterfactual.
            # Calculate Gt^{-m} for each module m
            for m_idx in range(self.num_modules):
                # Compute sum_im without module m
                sum_im_counterfactual = self._init_cls()
                for i in range(self.num_modules):
                    if i != m_idx:
                        sum_im_counterfactual += write_vectors_im[i] # Use the already generated Im
                sum_im_counterfactual = sum_im_counterfactual.coalesce()

                # Compute ct+1^{-m} using the same noise and gamma
                ct_plus_1_counterfactual = self.cls_update_rule(
                    self.ct, sum_im_counterfactual, mmod_t, gamma_t, noise_std_dev
                )

                # Compute Gt^{-m} by calculating surprises relative to ct+1^{-m}
                gt_counterfactual = 0.0
                for i, module in enumerate(self.modules):
                    # Surprise if module i's prediction (made based on ct) is compared to ct+1^{-m}
                    sm_counterfactual = F.mse_loss(predictions[i], ct_plus_1_counterfactual.to_dense(), reduction='mean')
                    gt_counterfactual += sm_counterfactual
                gt_counterfactual /= self.num_modules

                # Difference Reward Rm = Gt^{-m} - Gt
                rm = gt_counterfactual - global_surprise_gt
                # We want to *minimize* Gt, so reward should be negative of this?
                # Paper Prop 4.2 uses R = Gt^{-m} - Gt and aims to maximize R via policy gradient.
                # Gradient is approx E[R * grad(log(pi))]. Minimizing Gt is E[-Gt * grad(log(pi))].
                # So using R = Gt^{-m} - Gt directly in REINFORCE loss (-R*log_prob) should work.
                difference_rewards_rm.append(rm.detach()) # Detach reward from graph

        else: # First step, no baseline Gt? Or set R=0?
            difference_rewards_rm = [torch.tensor(0.0, device=self.device) for _ in range(self.num_modules)]

        # --- 7. Update State and Store ---
        self.ct = ct_plus_1
        self.last_gt = global_surprise_gt.detach()

        # --- 8. Trigger Learning ---
        module_losses = []
        for i, module in enumerate(self.modules):
            loss = module.learn(difference_rewards_rm[i])
            module_losses.append(loss)

        meta_loss = self.meta_model.learn() # Simplified meta-learning

        # --- 9. Return Metrics ---
        metrics = {
            "Gt": global_surprise_gt.item(),
            "Sm_avg": torch.mean(torch.stack(surprises_sm)).item(),
            "Sm_std": torch.std(torch.stack(surprises_sm)).item() if len(surprises_sm) > 1 else 0.0,
            "Rm_avg": torch.mean(torch.stack(difference_rewards_rm)).item(),
            "Rm_std": torch.std(torch.stack(difference_rewards_rm)).item() if len(difference_rewards_rm) > 1 else 0.0,
            "lambda_max": lambda_max, # Estimated lambda_max
            "gamma_t": gamma_t,
            "noise_std": noise_std_dev,
            "module_loss_avg": np.mean(module_losses) if module_losses else 0.0,
            "meta_loss": meta_loss,
            "cls_norm": torch.linalg.norm(self.ct.to_dense()).item(),
             "cls_density": self.ct.values().numel() / self.cls_dim if self.ct.values().numel() > 0 else 0.0
        }

        return metrics