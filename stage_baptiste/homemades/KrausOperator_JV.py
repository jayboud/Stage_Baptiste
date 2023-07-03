"""
KrausOperator_JV - (Jeremie's Version)
Author : Jeremie Boudreault
Date: 28/06/2023

Creating my version of the opListsBs2 function
in KrausOperator.py so it's compatible with
my GKP class (see finite_GKP.py) and thus can
be used for a qudit d.
"""
import numpy as np
import tqdm
import math
from qutip import *
from scipy.constants import pi
from stage_baptiste.homemades.finite_GKP import GKP
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator


def opListsBs2(GKP):
    """
    Function to create Kraus operators
    There are 2 stabilizers (x and p),
    and for each of them there are 2 measurement results (g and e)
    There are therefore 4 Kraus operators, which are returned as
    Klist = [[Kxg,Kxe],[Kpg,Kpe]]
    Measuring "g" means no errors, measuring "e" means one error

    Args:
        GKP: GKP object (from finite_GKP.py)
            The object containing the finite GKP state and
            it's useful characteristics (d,j,delta,hilbert_dim)
            to apply errors on.

    Returns:
        Klist: list
            The Kraus operators [[Kxg,Kxe],[Kpg,Kpe]]

    """

    d = GKP.d
    env = GKP.delta
    dim = GKP.hilbert_dim
    latticeGens = [np.sqrt(2*pi*d),1j*np.sqrt(2*pi*d)]
    Klist = []
    for i in range(2):
        latGen = latticeGens[i]
        dpie = displace(dim,1j*env*latGen/2/d/np.sqrt(2))
        dmie = dpie.dag()
        dpl = displace(dim,latGen/d/np.sqrt(2))
        dml = dpl.dag()
        deg = dpl*(dpie - 1j*dmie)/2
        dee = dml*(dpie + 1j*dmie)/2
        Klistg = (dpie*(deg + dee) + 1j*dmie*(deg - dee))/2
        Kliste = (dpie*(deg + dee) - 1j*dmie*(deg - dee))/2
        Klist.append([Klistg,Kliste])
        # Klist[2*i,1] = (dmie*(deg + dee) + 1im*dpie*(deg - dee))/2
        # Klist[2*i,2] = (dmie*(deg + dee) - 1im*dpie*(deg - dee))/2
    return Klist


def color_map(GKP_obj,H,t_gate,max_error_rate,max_N_rounds,fig_save_path,mode='fid',t_num=10,kap_num=10,
              rounds_steps=1,superposition_state=None,ss_hilbert_dim=None,ss_d=None,save=True,show=False):
    """
    Function that calculates and plot a fidelity or probability colormap
    Args:
        GKP_obj: GKP object or None if superposition_state
            The object containing everything relevant
            on the initial state to apply and correct errors on.
        H: Qobj (operator)
            The hamiltonian used to evolve the initial state.
        t_gate: float
            The time taken to apply the gate we want to study.
        max_error_rate: float
            The maximum error rate in units of t_gate.
        max_N_rounds: int
            The number of times to apply the error correcting
            procedure (one procedure includes x and p correction).
        fig_save_path: string
            The path and the name of the saved colormap.
        mode: string 'fid' or 'prob'
            Decides wether to get the fidelity or probability colormap.
        t_num: int
            The number of samples to generate in order to solve
            the evolution of the state under the gate.
        kap_num: int
            The number of samples to generate to compare error rates.
        rounds_steps: int
            The step between the number of error correcting rounds to compare.
        superposition_state: Qobj (ket)
            If the initial state is a superposition of many GKP.
        ss_hilbert_dim: int
            The hilbert dimension in Fock space of states in the superposition state.
        save: bool
            Saving the figure.
        show: bool
            Showing the figure.

    Returns a matplotlib figure of the desired colormap (fidelity or probability) with
    the error rate on the x axis and the number of error correcting rounds on the y axis.

    """
    if superposition_state:
        if GKP_obj:
            raise ValueError("You have to choose between a normal GKP state or a superposition.")
        if not ss_hilbert_dim:
            raise ValueError("You have to specify the hilbert dimension in Fock space of states"
                             "composing your superposition with the parameter 'ss_hilbert_dim'.")
        the_GKP = GKP()  # ---------------- #
        state = superposition_state
        dim = ss_hilbert_dim
    else:
        the_GKP = GKP_obj
        state = GKP.state
        dim = GKP.hilbert_dim
    t_list = np.linspace(0, t_gate, t_num)
    kap_max = max_error_rate/t_gate
    kap_list = np.linspace(0,kap_max,kap_num)
    rounds = np.arange(0,max_N_rounds,rounds_steps)
    a = destroy(dim)
    e_states = [mesolve(H,state,t_list,[np.sqrt(kap)*a],[]).states[-1] for kap in kap_list]
    opLists = opListsBs2(GKP)
    coin = np.random.random()
    if coin < 0.25:
        correction = 1.  # ---------- #
    return