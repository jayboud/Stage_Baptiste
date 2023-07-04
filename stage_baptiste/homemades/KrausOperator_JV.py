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


def get_correction_ix():
    """
    Function that randomly picks an
    index for a list of corrections of len=4.

    Returns: int
        The index.

    """
    correction_ix = None
    coin = np.random.random()
    if coin < 0.25:
        correction_ix = 0
    elif 0.25 <= coin < 0.5:
        correction_ix = 1
    elif 0.5 <= coin < 0.75:
        correction_ix = 2
    elif 0.75 <= coin < 1:
        correction_ix = 3
    elif coin == 1:
        get_correction_ix()
    return correction_ix


def color_map(GKP_obj,H,t_gate,max_error_rate,max_N_rounds,mode='fid',t_num=10,kap_num=10,N_rounds_steps=1,
              superposition_state=None,ss_d=None,ss_delta=None,ss_hilbert_dim=None,fig_name=None,fig_path=None,
              save=True,show=False):
    """
    Function that calculates and plot a fidelity or probability colormap of sBs error correction
    protocol applied to a state evolving under photon loss error sqrt(kappa)*a . The x and y axis
    of the colormap are respectively the dimensionless error rate kappa*tgate (kappa is in tgate units)
    and the number of error correcting rounds.
    Args:
        GKP_obj: GKP object (None if superposition_state)
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
        mode: string 'fid' or 'prob'
            Decides wether to get the fidelity or probability colormap.
        t_num: int
            The number of samples to generate in order to solve
            the evolution of the state under the gate.
        kap_num: int
            The number of samples to generate to compare error rates.
        N_rounds_steps: int
            The step between the number of error correcting rounds to compare.
        superposition_state: Qobj (ket)
            The initial state as a superposition of many GKP.states .
        ss_d: int (if superposition_state)
            The number d of logical states of the qudit space
            of states in the superposition.
        ss_delta: float (if superposition)
            The enveloppe on the finite stats of the superposition.
        ss_hilbert_dim: int (if superposition_state)
            The hilbert dimension in Fock space of states in the superposition state.
        fig_name: string
            The name under which to save the figure.
        fig_path: string
            The path and the name of the saved colormap.
        save: bool
            Saving the figure.
        show: bool
            Showing the figure.

    Creates a matplotlib figure of the desired colormap (fidelity or probability) with
    the error rate on the x axis and the number of error correcting rounds on the y axis.

    """
    if mode not in ['fid','prob']:
        raise ValueError("You have to choose a valid mode ('fid' or 'prob')")
    if superposition_state:
        if GKP_obj:
            raise ValueError("You have to choose between a normal GKP state or a superposition.")
        if not ss_d:
            raise ValueError("You have to specify the qudit d"
                             "composing your superposition with the kwarg 'ss_d'.")
        if not ss_delta:
            raise ValueError("You have to specify the enveloppe of states composing"
                             " your superposition with the kwarg 'ss_delta.")
        if not ss_hilbert_dim:
            raise ValueError("You have to specify the hilbert dimension in Fock space of states"
                             "composing your superposition with the kwarg 'ss_hilbert_dim'.")
        the_GKP = GKP(ss_d, 0, ss_delta, ss_hilbert_dim)  # creating a |0>_d state GKP object by default to put in opListsBs2
        state = superposition_state
        dim = ss_hilbert_dim

    else:
        the_GKP = GKP_obj
        state = the_GKP.state
        dim = the_GKP.hilbert_dim
    t_list = np.linspace(0, t_gate, t_num)
    kap_max = max_error_rate/t_gate
    kap_list = np.linspace(0,kap_max,kap_num)
    rate_list = np.linspace(0,max_error_rate,kap_num)
    N_rounds = np.arange(0,max_N_rounds,N_rounds_steps)
    a = destroy(dim)
    e_states = []
    for count,kap in enumerate(kap_list):
        options = Options(nsteps=2000)
        e_states.append(mesolve(-H,state,t_list,[np.sqrt(kap)*a],[],options=options).states[-1])
        print(f"e_states {count} done.")
    opList = opListsBs2(the_GKP)
    corrections = [opList[0][0]*opList[1][0],opList[0][0]*opList[1][1],  # [Bgg, Bge
                   opList[0][1]*opList[1][0],opList[0][1]*opList[1][1]]  # Beg, Bee]
    fidelities,probabilities = [],[]  # initializing lists
    for e_state in e_states:
        for N_round in N_rounds:
            psi = e_state
            for round in range(N_round):
                correction_ix = get_correction_ix()  # get correction at random
                correction = corrections[correction_ix]  # either Bgg,Bge,Beg or Bee
                psi = (correction*psi).unit()
            fidelities.append(fidelity(psi,state))
            # probabilities.append()  # ----- #
    fid_arr,prob_arr = np.array(fidelities),np.array(probabilities)

    # ***** plot *****
    xvec,yvec = rate_list,N_rounds
    fig,ax = plt.subplots()
    if mode == 'fid':
        cf = ax.pcolormesh(xvec,yvec,np.reshape(fid_arr,(len(kap_list),len(N_rounds))).T,cmap="seismic")
        fig.suptitle("Fidelity map")
    elif mode == 'prob':
        cf = ax.pcolormesh(xvec,yvec,np.reshape(prob_arr,(len(kap_list),len(N_rounds))).T,cmap="seismic")
        fig.suptitle("Probability map")
    fig.colorbar(cf,ax=ax)
    ax.set_xlabel(r"$\kappa t_{gate}$")
    ax.set_ylabel("N_rounds")
    if save:
        plt.savefig(fig_path+fig_name)
    if show:
        plt.show()
    return
