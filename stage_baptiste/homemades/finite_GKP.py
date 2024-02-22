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
from stage_baptiste.homemades.KrausOperators_original import opListsBs2


def get_d_gkp(d, j, delta, hilbert_dim, peak_range=10):
    """
    Function that calculates essentials for the creation of
    a finite qdit GKP state in regular space, such as his eigenvectors
    in the Fock basis with their respective coefficients,
    and the full normalized state.

    Args:
    d: int
        The dimension of codespace
    j: int
        The number of the logical state.
    delta: int of float
        The enveloppe of the finite GKP.
    hilbert_dim: int
        The number of dimensions of the Hilbert space (Fock space).
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
    full_state: qobj (ket)
        The GKP normalized state.
    """
    ns = np.arange(0,hilbert_dim,1)  # considering pair Fock states only
    js = np.arange(-peak_range, peak_range+1, 1)  # considering only the peaks in range of peak_range
    js = js + j/d
    xs = js*np.sqrt(2*pi*d)
    coeffs, eigen_states, states = [],[],[]  # creating the lists to return
    # calculating the coefficients c2n
    for n in ns:  # for each eigenstates
        herms = sum(np.exp(-xs*xs/2) * hermite(n)(xs))  # second term of multiplication
        enveloppe = np.exp(-delta**2 * n)  # enveloppe
        numerator = enveloppe*herms
        coeff = numerator/np.power(2,n/2,dtype='float')  # dividing by pieces (2**n)
        coeff /= np.sqrt(float(math.factorial(int(n))))  # dividing by pieces (factorial)
        coeff /= pi**(1/4)  # dividing by pieces (sqrt(pi))
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
    def __init__(self, d, j, delta, hilbert_dim):
        """

        Args:
            d: int
                The dimension of the codespace
            j: int (< d)
                The number of the logical state.
            delta: float
                The enveloppe of the finite state.
            hilbert_dim: int
                The number of dimensions of the Hilbert space (Fock space).
        """
        self.d = d
        self.j = j
        self.delta = delta
        self.hilbert_dim = hilbert_dim
        self.coeffs, self.eigen_states, self.state = get_d_gkp(self.d, self.j, self.delta, self.hilbert_dim)


class KrausGKP:
    """
    A class for a GKP state.
    """
    def __init__(self, d, j, delta, hilbert_dim):
        """

        Args:
            d: int
                The dimension of the codespace
            j: int (0 or 1)
                The number of the logical state.
            delta: float
                The enveloppe of the finite state.
            hilbert_dim: int
                The number of dimensions of the Hilbert space (Fock space).
        """
        self.d = d
        self.j = j
        self.delta = delta
        self.hilbert_dim = hilbert_dim
        latticeGens = [np.sqrt(2 * pi * d), 1j * np.sqrt(2 * pi * d)]
        Klist = opListsBs2(delta,latticeGens)
        Kgg = Klist[0][0]*Klist[1][0]
        herm_op = Kgg.dag()*Kgg
        eig_vals = herm_op.eigenstates()[0][-2:]
        eig_vects = herm_op.eigenstates()[1][-2:]
        self.coeffs, self.eigen_states, self.state = "This is a Kraus state", "This is a Kraus state", eig_vects[j]
