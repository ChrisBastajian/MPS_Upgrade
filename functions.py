import numpy as np


def neumann_self_inductance(x, y, z, wire_radius, Y):
    """
    Compute self-inductance of a loop using Neumann's formula.

    Parameters:
    x, y, z : ndarray
        Arrays of coordinates describing a closed loop.
    wire_radius : float
        Radius of the current-carrying wire.
    Y : float
        Skin effect correction factor (0 to 0.5).

    Returns:
    L_corrected : float
        Corrected self-inductance accounting for skin effect.
    L_uncorrected : float
        Uncorrected self-inductance assuming uniform current.
    """
    mu_0 = 4 * np.pi * 1e-7
    a = wire_radius

    num_segments = len(x) - 1
    ds = np.zeros((num_segments, 3))
    centers = np.zeros((num_segments, 3))
    starts = np.zeros((num_segments, 3))
    segment_lengths = np.zeros(num_segments)

    # Compute segments
    for i in range(num_segments):
        ds[i] = [x[i + 1] - x[i], y[i + 1] - y[i], z[i + 1] - z[i]]
        segment_lengths[i] = np.linalg.norm(ds[i])
        centers[i] = [(x[i + 1] + x[i]) / 2, (y[i + 1] + y[i]) / 2, (z[i + 1] + z[i]) / 2]
        starts[i] = [x[i], y[i], z[i]]

    wire_length = np.sum(segment_lengths)
    L_local = np.zeros(num_segments)

    # Neumann integral approximation
    for i in range(num_segments):
        for j in range(num_segments):
            R = np.linalg.norm(centers[i] - centers[j])
            if R > a / 2:
                dot_product = np.dot(ds[i], ds[j])
                L_ij = dot_product / R
                if i == j:
                    L_local[i] += L_ij  # Self-pair counted once
                else:
                    L_local[i] += L_ij  # Mutual pairs still counted once here

    L_sum = np.sum(L_local)
    L_corrected = mu_0 / np.pi * ((1 / 4) * L_sum + 0.5 * wire_length * Y)
    L_uncorrected = mu_0 * L_sum / (4 * np.pi)

    return L_corrected, L_uncorrected