import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize


# 1. Simulate CEV process data
def simulate_cev(sigma, beta, T, N, f0=1):
    dt = T / N
    f = np.zeros(N)
    f[0] = f0
    np.random.seed(21)

    for i in range(1, N):
        drift_term = -0.5 * sigma ** 2 * (f[i - 1]/ f0) ** (2 * beta) * dt
        diffusion_term = sigma * (f[i - 1]/ f0) ** beta * np.sqrt(dt) * np.random.randn()
        f[i] = f[i - 1] * np.exp(drift_term + diffusion_term)
        f[i] = max(f[i], 1e-10)
    return f, dt


# 2. Discretize data into bins
def discretize_data(data, num_bins):
    x_min, x_max = np.min(data), np.max(data)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)
    return indices, bins


# 3. Construct empirical Q matrix (not used)
def build_Q_matrix(state_seq, num_states, dt):
    N_ij = np.zeros((num_states, num_states))
    T_i = np.zeros(num_states)

    for i in range(len(state_seq) - 1):
        cur, nxt = state_seq[i], state_seq[i + 1]
        N_ij[cur, nxt] += dt
        T_i[cur] += dt

    Q = np.zeros_like(N_ij)
    for i in range(num_states):
        if T_i[i] > 0:
            Q[i] = N_ij[i] / T_i[i]
            Q[i, i] = -np.sum(Q[i]) + Q[i, i]
    return Q


# 4. Construct theoretical Q matrix(CEV process)
def theoretical_Q(grid, sigma, beta,f0):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))

    for i, x in enumerate(grid):
        if x <= 0:
            continue
        common_factor = -(grid[i] / f0) ** (2 * beta) * grid[i] ** 2 * sigma ** 2

        if i > 0:
            if i < N - 1:
                denominator =  grid[i - 1] * grid[i + 1] + grid[i] * grid[i - 1] - grid[i - 1]**2 - grid[i + 1] * grid[i]
            else:
                denominator = 2 * h ** 2
            if abs(denominator) > 1e-12:
                Q[i, i - 1] = max(common_factor / denominator, 0)

        if i < N - 1:
            if i > 0:  # Interior point
                denominator = (grid[i + 1] - grid[i - 1]) * (grid[i] - grid[i + 1])
            else:
                denominator = 2 * h ** 2
            if abs(denominator) > 1e-12:
                Q[i, i + 1] = max(common_factor / denominator, 0)

        Q[i, i] = -np.sum(Q[i])

    return Q


# 5. Log-likelihood function
def log_likelihood(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0

    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]
        prob = P[s1, s2]
        loglik += np.log(prob + eps)
    return -loglik


# 6. Fit parameters using likelihood
def fit_params_matrix(state_seq, bins, dt,f0):
    centers = (bins[:-1] + bins[1:]) / 2

    def loss(params):
        sigma, beta = params
        Q = theoretical_Q(centers, sigma, beta,f0)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(
        loss,
        x0=[0.2, -0.5],
        bounds=[(0.01, 0.99), (-2.0, 1.0)]
    )
    return res.x



# Main function
def main():
    # True parameters
    sigma_true, beta_true = 0.2, - 0.6
    T = 10
    N = 520
    f0 = 1

    # Simulate data
    data, dt = simulate_cev(sigma=sigma_true, beta=beta_true, T=T, N=N)

    # Discretize state space
    state_seq, bins = discretize_data(data, num_bins=5)
    print("Discrete state sequence:", state_seq)
    print("State intervals:", bins)

    # Construct theoretical Q matrix (for reference, not used in new loss function)
    centers = (bins[:-1] + bins[1:]) / 2
    Q_theo = theoretical_Q(centers, sigma_true, beta_true,f0)
    print("Theoretical Q matrix:\n", Q_theo)

    # Fit parameters using new loss function
    est_sigma, est_beta = fit_params_matrix(state_seq, bins, dt, f0)

    print(f"True parameters: σ = {sigma_true}, β = {beta_true}")
    print(f"Estimated parameters: σ̂ = {est_sigma:.4f}, β̂ = {est_beta:.4f}")

    # Plot CEV process trajectory and discrete state sequence
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(data)) * dt, data, label='CEV Process', color='blue', alpha=0.6)

    # Map state indices to values in bins
    state_value = np.array([bins[s] for s in state_seq])
    plt.step(np.arange(len(data)) * dt, state_value, label='Markov Process', color='red', alpha=0.6)

    plt.title('CEV Process Trajectory with Discrete State Space')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Parameter estimation errors
    param_true = np.array([sigma_true, beta_true])
    param_est = np.array([est_sigma, est_beta])
    mae = np.mean(np.abs(param_true - param_est))
    mse = np.mean((param_true - param_est) ** 2)

    print("\nError Analysis:")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"MSE (Mean Squared Error): {mse:.6f}")


if __name__ == "__main__":
    main()