"""
Author : Jeremie Boudreault
Date: 26/05/2023

Creating my own functions or modifying qutip functions.
"""

import numpy as np
from qutip.wigner import _wig_laguerre_val, _csr_get_diag
from qutip import *
from stage_baptiste.homemades.finite_GKP import get_d_gkp, GKP
from scipy.constants import pi


def rot_wigner_clenshaw(rho, xvec, yvec, rot=0,g=np.sqrt(2), sparse=False):
    r"""
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.

    ** I added the clockwise rotation of the meshgrid ** -Jay

    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / \sqrt(L!)` where
    :math:`c_L = \sum_n \rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    """

    M = np.prod(rho.shape[0])
    x,y = np.meshgrid(xvec, yvec)
    RotMatrix = np.array([[np.cos(rot), np.sin(rot)],
                          [-np.sin(rot), np.cos(rot)]])
    X,Y = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))
    #A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (X + 1.0j * Y) #this is A2 = 2*A

    B = np.abs(A2)
    B *= B
    w0 = (2*rho.data[0,-1])*np.ones_like(A2)
    L = M-1
    #calculation of \sum_{L} c_L (2x)^L / \sqrt(L!)
    #using Horner's method
    if not sparse:
        rho = rho.full() * (2*np.ones((M,M)) - np.diag(np.ones(M)))
        while L > 0:
            L -= 1
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    else:
        while L > 0:
            L -= 1
            diag = _csr_get_diag(rho.data.data,rho.data.indices,
                                rho.data.indptr,L)
            if L != 0:
                diag *= 2
            #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
            w0 = _wig_laguerre_val(L, B, diag) + w0 * A2 * (L+1)**-0.5

    return w0.real * np.exp(-B*0.5) * (g*g*0.5 / pi)


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
