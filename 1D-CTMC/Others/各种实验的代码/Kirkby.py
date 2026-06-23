import numpy as np

def Q_Matrix(m_0, mu_func, sig_func, lx, ux, gridMethod, center, GridMultParam):
    # Initialize variables
    c_index = -1  # center index, will be updated for non-uniform grids
    Q = np.zeros((m_0, m_0))

    if gridMethod == 1:  # Uniform grid
        nx = m_0
        dx = (ux - lx) / nx
        v = lx + np.arange(1, nx + 1) * dx

    elif gridMethod == 5 or gridMethod == 8:  # Tavella and Randall method
        tol = 1e-6
        v = np.zeros(m_0)
        v[0] = lx
        v[m_0 - 1] = ux
        alpha = GridMultParam * (v[m_0 - 1] - v[0])
        c1 = np.arcsinh((v[0] - center) / alpha)
        c2 = np.arcsinh((v[m_0 - 1] - center) / alpha)
        vtil = np.zeros(m_0 - 1)
        vtil[1:m_0 - 1] = center + alpha * np.sinh(
            c2 / (m_0 - 1) * np.arange(2, m_0) + c1 * (1 - np.arange(2, m_0) / (m_0 - 1)))

        # Find the index of the center point
        nnot_til = 2
        while vtil[nnot_til] < center:
            nnot_til += 1
        nnot_til -= 1
        v[1:nnot_til] = vtil[1:nnot_til]
        v[nnot_til + 1: m_0 - 1] = vtil[nnot_til + 1: m_0 - 2]

        # Check if center is too close to existing points
        if center - vtil[nnot_til] < tol:
            c_index = nnot_til
            v[c_index] = center
            v[nnot_til + 1] = (center + vtil[nnot_til + 1]) / 2
        elif vtil[nnot_til + 1] - center < tol:
            c_index = nnot_til + 2
            v[c_index] = center
            v[nnot_til + 2] = (v[nnot_til + 2] + v[nnot_til]) / 2
        else:
            c_index = nnot_til + 1
            v[c_index] = center

    elif gridMethod == 7:  # Another non-uniform grid method
        v = np.zeros(m_0)
        v[m_0 - 1] = ux
        v[0] = lx
        alpha = GridMultParam * (ux - lx)
        c1 = np.arcsinh((lx - center) / alpha)
        c2 = np.arcsinh((ux - center) / alpha)
        v[1:m_0 - 1] = center + alpha * np.sinh(c2 / m_0 * np.arange(2, m_0) + c1 * (1 - np.arange(2, m_0) / m_0))

        c_index = 1
        while v[c_index] < center:
            c_index += 1

        if center != 0:
            ratio = center / v[c_index]
            v = v * ratio  # Scale grid to center
        else:
            v = v + center - v[c_index]

    # Now Generate Q Matrix
    mu_vec = mu_func(v)
    mu_plus = np.maximum(0, mu_vec)
    mu_minus = np.maximum(0, -mu_vec)
    sig2 = sig_func(v) ** 2

    if gridMethod == 1:  # Uniform grid
        for i in range(1, m_0 - 1):
            temp = np.maximum(sig2[i] - dx * (mu_minus[i] + mu_plus[i]), 0) / (2 * dx ** 2)
            Q[i, i - 1] = mu_minus[i] / dx + temp  # j = i-1
            Q[i, i + 1] = mu_plus[i] / dx + temp  # j = i+1
            Q[i, i] = -Q[i, i - 1] - Q[i, i + 1]  # j = i

        Q[0, 1] = abs(mu_vec[0]) / dx
        Q[m_0 - 1, m_0 - 2] = abs(mu_vec[m_0 - 1]) / dx

    elif gridMethod == 8:  # Another non-uniform grid method
        H = np.diff(v)
        HD = H[0]
        HU = H[0]
        AA = np.maximum(sig2[0] - (HU * mu_plus[0] + HD * mu_minus[0]), 0) / (HU + HD)
        Q[0, 1] = (mu_plus[0] + AA) / HU
        Q[0, 0] = -Q[0, 1]

        HD = H[m_0 - 2]
        HU = H[m_0 - 2]
        AA = np.maximum(sig2[m_0 - 1] - (HU * mu_plus[m_0 - 1] + HD * mu_minus[m_0 - 1]), 0) / (HU + HD)
        Q[m_0 - 1, m_0 - 2] = (mu_plus[m_0 - 1] + AA) / HU
        Q[m_0 - 1, m_0 - 1] = -Q[m_0 - 1, m_0 - 2]

        for i in range(1, m_0 - 1):
            dvU = v[i - 1] - v[i]
            dvD = v[i + 1] - v[i]
            C = np.array([[1, 1, 1], [dvU, 0, dvD], [dvU ** 2, 0, dvD ** 2]])
            z = np.array([0, mu_vec[i], sig2[i]])
            hrow = np.linalg.solve(C, z)
            Q[i, i - 1] = hrow[0]
            Q[i, i] = hrow[1]
            Q[i, i + 1] = hrow[2]

    else:  # Non-uniform grids
        H = np.diff(v)
        for i in range(1, m_0 - 1):
            HD = H[i - 1]  # Down step
            HU = H[i]  # Up step
            AA = np.maximum(sig2[i] - (HU * mu_plus[i] + HD * mu_minus[i]), 0) / (HU + HD)
            Q[i, i - 1] = (mu_minus[i] + AA) / HD
            Q[i, i + 1] = (mu_plus[i] + AA) / HU
            Q[i, i] = -Q[i, i - 1] - Q[i, i + 1]

        # Boundary behavior
        HD = H[0]
        HU = H[0]
        AA = np.maximum(sig2[0] - (HU * mu_plus[0] + HD * mu_minus[0]), 0) / (HU + HD)
        Q[0, 1] = (mu_plus[0] + AA) / HU

        HD = H[m_0 - 2]
        HU = H[m_0 - 2]
        AA = np.maximum(sig2[m_0 - 1] - (HU * mu_plus[m_0 - 1] + HD * mu_minus[m_0 - 1]), 0) / (HU + HD)
        Q[m_0 - 1, m_0 - 2] = (mu_plus[m_0 - 1] + AA) / HU

    Q[0, 0] = -Q[0, 1]
    Q[m_0 - 1, m_0 - 1] = -Q[m_0 - 1, m_0 - 2]

    return Q, v, c_index

    print("Grid v:")
    print(v)
    print("\nQ Matrix:")
    print(Q)
    print("\nCenter Index:")
    print(c_index)