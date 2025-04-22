The user wants me to implement the changes outlined in the "Revised Plan for Phase 1 Revision 3" into the provided Python code files (`core.py`, `train.py`, `envs.py`, `utils.py`).

**Plan Checklist & Implementation Details:**

1.  **Attack Exploding Gradients:**
    *   **Increase Action Std:** Modify `DEFAULT_ACTION_STD` in `core.py` and ensure `train.py` uses the `action_std` argument when initializing modules. *Done.*
    *   **Add Weight Decay:** Add `weight_decay` argument to `train.py` parser. Modify `PredictiveModule`, `TaskHead`, and `MetaModel` `__init__` methods in `core.py` to accept `weight_decay` and pass it to their `Adam` optimizers. *Done.*
    *   **Gradient Clipping:** Already implemented (`GRAD_CLIP_NORM = 1.0`). Keep monitoring. *No code change needed.*
    *   **Activation Clipping:** Defer this unless necessary. *No code change needed.*

2.  **Revisit Difference Reward (Option B - Approx Counterfactual):**
    *   Modify `DPLDSystem` in `core.py`:
        *   Add `self.last_log_surprises_sm = None` to `__init__`.
        *   In `step()` method, after calculating `log_surprises_sm`, store this list in `self.last_log_surprises_sm`.
        *   Modify the reward calculation loop:
            *   Check if `self.last_log_surprises_sm` exists (handle first step).
            *   For module `i`, calculate the baseline `baseline_m_prev` by averaging `self.last_log_surprises_sm[j]` for all `j != i`. Handle potential NaNs/Infs in previous surprises.
            *   Calculate `Rm_log = baseline_m_prev - global_log_surprise_gt.item()`.
            *   Normalize these `Rm_log` values across modules.
            *   Pass the normalized reward to `module.learn()`.
    *   *Done.*

3.  **Investigate and Calibrate LE Estimation:**
    *   **Dynamics Map:** Defer changes. *No code change needed.*
    *   **Meta-Model Stability Target:** Modify default value in `train.py` parser for `--stability-target`. *Done.*

4.  **Tune Hyperparameters (Defaults in `train.py`):**
    *   **Action Std:** Update default in `train.py`. *Done.*
    *   **Weight Decay:** Add argument and default in `train.py`. *Done.*
    *   **Learning Rates:** Keep defaults low. *No code change needed.*
    *   **Entropy Coeff:** Update default in `train.py`. *Done.*
    *   **Task Loss Weight:** Keep default low (`0.2`). *No code change needed.*

5.  **Code Implementation:** Generate the full code for `core.py` and `train.py` with these modifications. `envs.py` and `utils.py` likely don't need changes for this revision.

**Reviewing Papers for Stability Inspiration:**

*   **Part I, Sec 4.0.2 (Ensuring Stability):** Mentions intrinsic homeostasis (decay, normalization), Meta-Model regulation, local constraints (regularization - *we added weight decay*), balanced plasticity (managed LRs - *we lowered them*), noise injection (*we have annealing*), structural priors (*architecture itself*).
*   **Part II, Sec 7 (Stability Guarantees):** Mentions Lipschitz bounding (via spectral norm - *deferred*), adaptive decay (`gamma_t` via Meta-Model - *implemented*), empirical Lyapunov monitoring (*implemented, needs calibration*), Stability Guard via adaptive LR (Theorem 7.1 - *not implemented, relies on LE accuracy*), noise scheduling (*implemented*).
*   **Part II, Sec 11 (Limitations):** Mentions algorithmic specification and tuning, scalability, biological gaps, evaluation bottlenecks.

The papers anticipate stability challenges. Our current approach aligns with the proposed mechanisms: noise, decay regulation, LE monitoring (though calibration needed), regularization (weight decay), and managed learning rates. The simplified DR is a pragmatic step to bypass the complexity of the full counterfactual mentioned but not fully detailed algorithmically in the papers. The extremely large gradients suggest a fundamental issue perhaps related to the interaction between the surprise calculation (especially log-surprise on potentially large raw values), the action sampling variance (`action_std`), and the network parameter scale. Increasing `action_std` and adding `weight_decay` are direct attempts to address these potential interactions.