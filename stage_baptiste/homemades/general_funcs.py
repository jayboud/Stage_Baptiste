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
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import wraps


def cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_args, last_kwargs, last_outputs = wrapper._cache
        nb_args_match = len(args) == len(last_args)
        keys_match = kwargs.keys() == last_kwargs.keys()
        if nb_args_match and keys_match:
            args_match = all([np.all(np.equal(a_new, a_old)) for a_new, a_old in zip(args, last_args)])
            kwargs_match = all([np.all(np.equal(kwargs[key], arg)) for key, arg in kwargs.items()])
            if last_outputs is not None and args_match and kwargs_match:
                return last_outputs
        wrapper._cache = [args, kwargs, func(*args)]
        return wrapper._cache[-1]
    wrapper._cache = [tuple(), dict(), None]
    return wrapper


def chi_function(rho,z_max):
    """
    ****** not completed *******
    Args:
        rho: ndarray
            The density matrix.
        z_max:  float (real)
            The maximum phase to calculate it"s
            displacement over one dimension.

    Returns:
        chi_s: list of ndarrays
            The characteristic function chi_s(z,z^*) and xvec


    """
    rho_arr = np.array(rho)[:,:,None,None]
    xvec = np.linspace(-z_max,z_max,200)
    x,y = np.meshgrid(xvec,xvec)
    z = (x+1j*y)[None,None]
    zc = np.conj(z)
    print(z,zc)
    dimension = rho.shape[0]
    a = destroy(dimension)
    a_ar, adag_ar = [np.array(op)[:,:,None,None] for op in [a,a.dag()]]
    chi_s = np.trace(rho_arr*np.exp(1j*zc*adag_ar + 1j*z*a_ar))
    return [chi_s,xvec]


def plot_chi(chi_l):
    """

    Args:
        fig: Figure object
            The figure on which to plot.
        ax: Axes object
            The ax on which to plot.
        chi_l: list of ndarrays
            The characteristic function and xvec.


    Returns:
    """
    fig,axs = plt.subplots(1,2,figsize=(8,4))
    chi_s,xvec = chi_l
    chi_lim = abs(chi_s).max()
    axs[0].contourf(xvec, xvec, np.real(chi_s), 100,norm=mpl.colors.Normalize(-chi_lim, chi_lim),cmap=mpl.colormaps['seismic'])
    axs[0].set_title(r"$\Re[\chi_s]$")
    axs[1].contourf(xvec, xvec, np.imag(chi_s), 100,norm=mpl.colors.Normalize(-chi_lim, chi_lim),cmap=mpl.colormaps['seismic'])
    axs[1].set_title(r"$\Im[\chi_s]$")
    return


def rot_wigner_clenshaw(rho, xvec, yvec, rot=0,g=np.sqrt(2), sparse=False):
    """
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.

    ** I added the anticlockwise rotation of the meshgrid ** -Jay

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



