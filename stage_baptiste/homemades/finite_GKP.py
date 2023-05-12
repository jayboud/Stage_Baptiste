"""
Author : Jeremie Boudreault
Date: 10/05/2022

Creating a class for finite GKP states, according
to document Analytical_GKP_in_Fock_basis.pdf
"""
import numpy as np
import qutip as qt
import math
from scipy.special import hermite
from scipy.constants import pi


def get_gkp(delta, hilbert_dim, peak_range=10):
    """
    Function that calculates essentials for the creation of
    a finite GKP state in regular space, such as his eigenvectors
    in the Fock basis with their respective coefficients,
    and the full normalized state.

    Parameters:
    delta: int of float
        The index of the GKP state in regular space.
    hilbert_dim: int
        The number of dimensions in the Hilbert space
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
    ns = np.arange(0,hilbert_dim,1)  # considering eigenstates
    js = np.arange(1, peak_range+1, 1)  # considering only the peaks in range of peak_range
    coeffs, eigen_states, states = [],[],[]  # creating the lists to return
    crit = 60
    # calculating the coefficients c2n
    for n in ns:  # for each eigenstates
        enveloppe = np.exp(-delta**2 * 2 * n)  # enveloppe
        herms = hermite(2*n)(0) + 2*sum(
            np.exp(-js*js*2*pi) * hermite(2*n)(js*2*np.sqrt(pi)))  # second term of multiplication
        numerator = enveloppe*herms
        if n > 63:
            bns = np.arange(crit+1,n+1,1)  # taking out big ns over crit value to avoid overflow in factorial
            coeff = numerator/np.power(2,n,dtype='float')  # dividing by pieces (2**n)
            coeff /= np.sqrt(float(math.factorial(2*crit)))  # dividing by pieces (first part of factorial)
            for bn in bns:
                coeff /= np.sqrt(float(bn))  # dividing by pieces (second part of factorial)
        else:
            coeff = numerator/(np.power(2,n,dtype='float')/np.sqrt(float(math.factorial(2*int(n)))))  # first term in multiplication
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
    def __init__(self, delta, hilbert_dim):
        self.delta = delta
        self.hilbert_dim = hilbert_dim
        self.coeffs, self.eigen_states, self.state = get_gkp(self.delta, self.hilbert_dim)
