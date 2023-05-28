"""
Author : Jeremie Boudreault
Date: 10/05/2023

Creating a class for finite GKP states, according
to document Analytical_GKP_in_Fock_basis.pdf
"""
import numpy as np
import qutip as qt
import math
from scipy.special import hermite
from scipy.constants import pi


def get_d_gkp(delta, hilbert_dim, m, k, peak_range=10):
    """
    Function that calculates essentials for the creation of
    a finite qdit GKP state in regular space, such as his eigenvectors
    in the Fock basis with their respective coefficients,
    and the full normalized state.

    Parameters:
    delta: int of float
        The index of the GKP state in regular space.
    hilbert_dim: int
        The number of dimensions of the Hilbert space.
    m : int
    The gap between considered Fock states.
    k:
    The starting number of considered Fock states
    peak_range: int
        The number of peaks (at x>=0) that we want to
        consider, taken to be 10 by default.

    Returns:
    coeffs: list of floats
        The coefficients before each eigenvectors
        in Fock space.
    eigen_states: list of kets
        The pair eigenstates in Fock space necessary
        to compute GKP in regular space.
    full_state: ket
        The GKP normalized state.
    """
    ns = np.arange(k,hilbert_dim,m)  # considering mod d eigenstates only
    js = np.arange(1, peak_range+1, 1)  # considering only the peaks in range of peak_range
    coeffs, eigen_states, states = [],[],[]  # creating the lists to return
    # calculating the coefficients c2n
    for n in ns:  # for each eigenstates
        """
        Note: In the proceeding calculations, 2n is replaced
              by n since we're already considering even states in ns.
        """
        enveloppe = np.exp(-delta**2 * n)  # enveloppe
        herms = hermite(n)(0) + 2*sum(
            np.exp(-js*js*2*pi) * hermite(n)(js*2*np.sqrt(pi)))  # second term of multiplication
        numerator = enveloppe*herms
        coeff = numerator/np.power(2,n/2,dtype='float')  # dividing by pieces (2**n)
        coeff /= np.sqrt(float(math.factorial(int(n))))  # dividing by pieces (factorial)
        eigen_state = qt.fock(hilbert_dim, n)  # creating eigen_state
        state = coeff*eigen_state  # eigenstate weighted by coefficient
        coeffs.append(coeff)
        eigen_states.append(eigen_state)
        states.append(state)
    full_state = sum(states).unit()  # getting GKP state by summing on all weighted eigenstates and normalizing
    return coeffs, eigen_states, full_state


class GKP:
    """
    A class for a GKP state.
    """
    def __init__(self, delta, hilbert_dim, m, k):
        self.delta = delta
        self.hilbert_dim = hilbert_dim
        self.mod = m
        self.first = k
        self.coeffs, self.eigen_states, self.state = get_d_gkp(self.delta, self.hilbert_dim, self.mod, self.first)
