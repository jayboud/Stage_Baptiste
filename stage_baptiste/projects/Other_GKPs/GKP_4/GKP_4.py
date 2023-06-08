"""
Author : Jeremie Boudreault
Date: 11/05/2023

Making calculation for a GKP code with d=4.
"""

import numpy as np
import scipy as sp
from scipy.constants import pi


def DFT_matrix(N):
    """
    Calculating the discrete Fourier transform matrix
    Args:
        N: float
            The number of dimensions.

    Returns:
        W: ndarray
        The DFT matrix.

    """
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(2*pi*1j/N)
    W = np.power(omega, i*j) / np.sqrt(N)
    return W

# Calculating eigenstates and eigenvalues of
# the 8x8 discrete Fourier matrix.


N = 8
F = DFT_matrix(N)  # matrix
print(np.linalg.eig(F)[1])
