import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize


# 1. Simulate OU process data
def simulate_ou(theta, sigma, mu=0.0, x0=0, T=50, N=1000, seed=None):
    dt = T / N
    x = np.zeros(N)
    x[0] = x0
    np.random.seed(seed)  # Use the specified seed
    for i in range(1, N):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x, dt


# 2. Discretize data into states (bins)
def discretize_data(data, num_bins):
    x_min, x_max = np.min(data), np.max(data)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)
    return indices, bins


# 3. Compute N_ij and T_i, construct empirical Q matrix
def build_Q_matrix(state_seq, num_states, dt):
    N_ij = np.zeros((num_states, num_states))
    T_i = np.zeros(num_states)
    for i in range(len(state_seq) - 1):
        cur_state = state_seq[i]
        next_state = state_seq[i + 1]
        N_ij[cur_state, next_state] += dt
        T_i[cur_state] += dt
    Q = np.zeros_like(N_ij)
    for i in range(num_states):
        if T_i[i] > 0:
            Q[i] = N_ij[i] / T_i[i]
            Q[i, i] = -np.sum(Q[i]) + Q[i, i]
    return Q


# 4. Construct the theoretical Q matrix for the OU model (for fitting)
def theoretical_Q(grid, theta, mu, sigma):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))
    for i, x in enumerate(grid):
        drift = theta * (mu - x)
        diff = sigma ** 2
        if i > 0:
            Q[i, i - 1] = max(diff / (2 * h ** 2) - drift / (2 * h), 0)
        if i < N - 1:
            Q[i, i + 1] = max(diff / (2 * h ** 2) + drift / (2 * h), 0)
        Q[i, i] = -np.sum(Q[i])
    return Q


# 5. Compute the MLE loss function
def log_likelihood(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0
    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]
        prob = P[s1, s2]
        loglik += np.log(prob + eps)
    return -loglik  # The negative sign is for minimization


# 6. Fit the OU parameters (maximize likelihood)
def fit_params(state_seq, bins, dt):
    centers = (bins[:-1] + bins[1:]) / 2

    def loss(params):
        theta, sigma = params
        Q = theoretical_Q(centers, theta, 0.0, sigma)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(loss, x0=[0.5, 0.2], bounds=[(0.01, 5), (0.01, 3)])
    return res.x


# Main program
def main():
    theta_true, sigma_true = 1.0, 0.3
    seeds = [3, 23, 50, 2024, 1000]
    bin_list = [10, 30, 50, 100, 150, 250]

    est_thetas_all_bins = []
    est_sigmas_all_bins = []

    print("Error analysis results:")

    # For each bin count in bin_list, compute parameter estimates for different seeds
    for num_bins in bin_list:
        seed_est_thetas = []
        seed_est_sigmas = []

        for seed in seeds:
            # Simulate data for each seed, discretize data, and fit parameters
            data, dt = simulate_ou(theta=theta_true, sigma=sigma_true, seed=seed)
            state_seq, bins = discretize_data(data, num_bins)
            est_theta, est_sigma = fit_params(state_seq, bins, dt)

            seed_est_thetas.append(est_theta)
            seed_est_sigmas.append(est_sigma)

            mae_theta = abs(est_theta - theta_true)
            mse_theta = (est_theta - theta_true) ** 2
            mae_sigma = abs(est_sigma - sigma_true)
            mse_sigma = (est_sigma - sigma_true) ** 2

            print(f"\nSeed = {seed}, num_bins = {num_bins}")
            print(f"Estimated parameters: θ = {est_theta:.4f}, σ = {est_sigma:.4f}")
            print(f"θ MAE: {mae_theta:.4f}, θ MSE: {mse_theta:.6f}")
            print(f"σ MAE: {mae_sigma:.4f}, σ MSE: {mse_sigma:.6f}")

        # Calculate average estimates for each bin count
        avg_theta = np.mean(seed_est_thetas)
        avg_sigma = np.mean(seed_est_sigmas)
        est_thetas_all_bins.append(avg_theta)
        est_sigmas_all_bins.append(avg_sigma)

        print(f"\nAverage estimates for num_bins = {num_bins}: θ = {avg_theta:.4f}, σ = {avg_sigma:.4f}")

    # Plot comparison of estimated parameters
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(bin_list, est_thetas_all_bins, marker='o', label='Estimated θ')
    plt.axhline(y=theta_true, color='r', linestyle='--', label='True θ = 1.0')
    plt.title('θ vs. num_bins (Average values for multiple seeds)')
    plt.xlabel('num_bins')
    plt.ylabel('θ')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bin_list, est_sigmas_all_bins, marker='s', color='green', label='Estimated σ')
    plt.axhline(y=sigma_true, color='r', linestyle='--', label='True σ = 0.3')
    plt.title('σ vs. num_bins (Average values for multiple seeds)')
    plt.xlabel('num_bins')
    plt.ylabel('σ')
    plt.grid(True)
    plt.legend()

    plt.suptitle('Parameter Estimates vs. Number of Bins (Average values for multiple seeds)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
