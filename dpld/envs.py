import torch
import numpy as np

class LorenzEnv:
    """
    Simulates the Lorenz attractor dynamics.
    Provides states for the DPLD system to predict.
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01, device='cpu'):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.device = device
        self.state = None
        self.reset()

    def _lorenz_dynamics(self, state_tensor):
        x, y, z = state_tensor[..., 0], state_tensor[..., 1], state_tensor[..., 2]
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        # Stack along the last dimension
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)

    def step(self):
        """Advances the simulation by one time step using RK4 integration."""
        k1 = self._lorenz_dynamics(self.state)
        k2 = self._lorenz_dynamics(self.state + 0.5 * self.dt * k1)
        k3 = self._lorenz_dynamics(self.state + 0.5 * self.dt * k2)
        k4 = self._lorenz_dynamics(self.state + self.dt * k3)
        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.get_observation()

    def reset(self, initial_state=None):
        """Resets the environment state."""
        if initial_state is None:
            # Use standard initial conditions or random ones
            # self.state = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=self.device)
             self.state = (torch.rand(3, device=self.device) - 0.5) * 2 * 10 # Random start
        else:
            self.state = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        return self.get_observation()

    def get_observation(self):
        """Returns the current state."""
        # In this simple case, observation is the state itself.
        # Could add noise or projection later.
        return self.state.clone()

    def get_dimension(self):
        return 3

# Example usage:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LorenzEnv(device=device)
    print(f"Device: {env.device}")
    print(f"Initial state: {env.reset()}")
    trajectory = []
    for _ in range(1000):
        state = env.step()
        trajectory.append(state.cpu().numpy())

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_title("Lorenz Attractor Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()