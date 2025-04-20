---

## 🧠 Theoretical Foundations

This codebase directly reflects the following theoretical constructs:

- **Equation 1:** Sparse Euler-Maruyama CLS update  
- **Lemma 1:** Bounded state norm under decay and bounded inputs  
- **Algorithm 1:** Sparse module→CLS write with gating and surprise scaling  
- **Proposition 2:** Difference reward as credit assignment signal  
- **Algorithm 2:** Lyapunov exponent monitoring and learning rate scaling

See the [papers](#) for full mathematical descriptions.

---

## 🚧 Project Status

This repository is under active development.  
All code is experimental and serves as a platform for testing hypotheses proposed in the DPLD theory papers. Results will be compiled into a future **Phase 3 empirical validation paper**.

---

## 📊 Planned Experiments

| Phase | Environment     | Objective                                  | Key Metrics                |
|-------|------------------|--------------------------------------------|----------------------------|
| P0    | Lorenz-63        | Internal stability, λ_max, CLS norm        | λ_max ↓, Surprise ↓        |
| P1    | Pixel CartPole   | Coherent motor control, curiosity effects  | Episodic Reward ↑, Φ̂ ↑    |

---

## 🔗 Citations

If you use this code or framework, please cite:

```bibtex
@misc{landes2025dpld,
  author       = {Isaac Landes and Samuel Berkebile},
  title        = {Distributed Predictive Latent Dynamics: Initial Exploration and Formal Framework},
  year         = {2025},
  howpublished = {arXiv preprint},
  note         = {Parts I & II},
  url          = {https://arxiv.org/abs/25xx.xxxxx} (not yet published)
}