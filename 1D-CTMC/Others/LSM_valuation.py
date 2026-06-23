import numpy as np
from scipy.stats import norm


# 1. Simulate CEV process data
def simulate_cev(sigma, beta, T, M, f0=100):
    dt = T / M
    f = np.zeros(M)
    f[0] = f0
    np.random.seed(11)

    for i in range(1, M):
        f[i] = f[i - 1] + f[i - 1] * sigma * (f[i - 1] / f0) ** beta * np.sqrt(dt) * np.random.randn()
    return f, dt


# 2. LSM American option pricing using dynamic programming
def price_American_option_lsm(sigma, beta, f0, K, r, T, M, n_paths, option_type='put'):
    dt = T / M
    df = np.exp(-r * dt)

    # Initialize price paths matrix
    paths = np.zeros((n_paths, M + 1))
    paths[:, 0] = f0
    np.random.seed(21)

    # Generate CEV price paths
    for i in range(n_paths):
        for t in range(1, M + 1):
            if paths[i, t - 1] > 0:
                dW = np.sqrt(dt) * np.random.randn()
                paths[i, t] = paths[i, t - 1] + paths[i, t - 1] * sigma * (paths[i, t - 1] / f0) ** beta * dW
                paths[i, t] = max(paths[i, t], 0.01)  # Ensure positive price
            else:
                paths[i, t] = 0.01

    # Calculate intrinsic value matrix
    if option_type == 'put':
        payoff = np.maximum(K - paths, 0)
    else:
        payoff = np.maximum(paths - K, 0)

    # Initialize value matrix with terminal payoff
    V = payoff[:, -1].copy()

    # Backward induction using LSM
    for t in range(M - 1, 0, -1):
        # Current stock prices and intrinsic values
        S_t = paths[:, t]
        intrinsic = payoff[:, t]

        # Select in-the-money paths only
        itm = intrinsic > 0

        if np.sum(itm) >= 20:  # Need sufficient ITM paths for regression
            # Regression: fit continuation value as function of stock price
            X = S_t[itm]
            Y = V[itm] * df

            # Use polynomial regression (degree 2)
            try:
                coeffs = np.polyfit(X, Y, 2)
                continuation = np.polyval(coeffs, S_t)

                # Exercise decision: immediate vs continuation value
                exercise = (intrinsic > continuation) & itm
                V = np.where(exercise, intrinsic, V * df)

            except:
                # If regression fails, just discount
                V = V * df
        else:
            # Too few ITM paths, just discount
            V = V * df

    return np.mean(V)


def main():
    sigma_true = 0.16
    beta_true = - 0.6
    T = 12
    M = 1000  # Increase time steps for better accuracy
    f0 = 100
    K = 100
    r = 0.03

    print("=== CEV American option pricing using LSM ===")
    print(f"Parameters: σ={sigma_true}, β={beta_true}")
    print(f"Initial futures price: {f0}, Strike price: {K}")

    # Step 1: Simulate CEV process
    print("\nStep 1: Simulate CEV process...")
    f, dt = simulate_cev(sigma_true, beta_true, T, M + 1, f0)
    print(f"Simulated {len(f)} time points, time step dt: {dt:.4f}")

    # Step 2: LSM American put option pricing
    print("\nStep 2: LSM American put option pricing...")
    n_paths = 10000  # Increase paths for better accuracy
    put_price_lsm = price_American_option_lsm(sigma_true, beta_true, f0, K, r, T, M, n_paths, 'put')

    # Step 3: LSM American call option pricing
    print("\nStep 3: LSM American call option pricing...")
    call_price_lsm = price_American_option_lsm(sigma_true, beta_true, f0, K, r, T, M, n_paths, 'call')

    print(f"Put option price (LSM): {put_price_lsm:.6f}")
    print(f"Call option price (LSM): {call_price_lsm:.6f}")

    # Theoretical Black-Scholes prices for comparison
    d1 = (np.log(f0 / K) + (r + 0.5 * sigma_true ** 2) * T) / (sigma_true * np.sqrt(T))
    d2 = d1 - sigma_true * np.sqrt(T)

    bs_put = K * np.exp(-r * T) * norm.cdf(-d2) - f0 * norm.cdf(-d1)
    bs_call = f0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print(f"Put option (BS): {bs_put:.6f}")
    print(f"Call option (BS): {bs_call:.6f}")

    print(f"Put option pricing error: {abs(bs_put - put_price_lsm):.6f}")
    print(f"Call option pricing error: {abs(bs_call - call_price_lsm):.6f}")


if __name__ == "__main__":
    main()