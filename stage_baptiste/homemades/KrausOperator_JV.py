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


def error_prob(error,rho):
    """
    Function to calcultate the probability of an error to occur.
    Args:
        error: Qobj
            The error matrix.
        rho: Qobj
            The density matrix.

    Returns:
        The probability that this error occurs.

    """
    return (error*rho*error.dag()).tr()


def get_correction(Klist,rho):
    """
    Function that randomly picks an
    index for a list of corrections of len=4.
    Args:
        Klist: list of Qobj
            Operators used to apply sBs error correction.
        rho: Qobj (operator)
            The density matrix to correct.

    Returns: int
        The index.

    """
    corrections = [Klist[0][0] * Klist[1][0], Klist[0][0] * Klist[1][1],  # [Bgg, Bge
                   Klist[0][1] * Klist[1][0], Klist[0][1] * Klist[1][1]]  # Beg, Bee]
    error_probs = np.array([error_prob(correction,rho) for correction in corrections])
    error_probs /= np.sum(error_probs)  # normalizing probabilities
    coin = np.random.random()
    if coin < error_probs[0]:
        correct_with = corrections[0]
    elif error_probs[0] <= coin < np.sum(error_probs[:2]):
        correct_with = corrections[1]
    elif np.sum(error_probs[:2]) <= coin < np.sum(error_probs[:3]):
        correct_with = corrections[2]
    elif np.sum(error_probs[:3]) <= coin < 1:
        correct_with = corrections[3]
    elif coin == 1:
        get_correction(Klist,rho)
    return correct_with


def color_maps(GKP_obj,H,t_gate,max_error_rate,max_N_rounds,t_num=10,kap_num=10,N_rounds_steps=1,mode='random',superposition_state=None,
               ss_d=None,ss_delta=None,ss_hilbert_dim=None,fig_name=None,fig_path=None,save=True,show=False):
    """
    Function that calculates and plot a fidelity colormap (and a probability colormap if mode=gg) of sBs error correction
    protocol applied to a qubit state evolving under photon loss error sqrt(kappa)*a . The x and y axis
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
        t_num: int
            The number of samples to generate in order to solve
            the evolution of the state under the gate.
        kap_num: int
            The number of samples to generate to compare error rates.
        N_rounds_steps: int
            The step between the number of error correcting rounds to compare.
        mode: string ('random' or 'gg' or 'avg')
            Decides wether or not you correct randomly or only with Bgg detection.
            If only Bgg detection, a probability map will aslo be plotted.
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
    # dealing with a simple GKP state or a superposition of states.
    if mode not in ['random','gg','avg']:
        raise ValueError("You have to choose a mode between 'random' or 'gg' or 'avg.")
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
    # setting all good parameters
    Y = displace(dim, np.sqrt(pi/2)*(1+1j))  # X gate for a qubit
    t_list = np.linspace(0, t_gate, t_num)
    kap_max = max_error_rate/t_gate
    kap_list = np.linspace(0,kap_max,kap_num)
    rate_list = np.linspace(0,max_error_rate,kap_num)
    N_rounds = np.arange(0,max_N_rounds,N_rounds_steps)
    a = destroy(dim)
    options = Options(nsteps=2000)
    fid_rho = mesolve(-H, state, t_list, [], [], options=options).states[-1]  # reference state for fidelity (no c_ops)
    norm_fid_rho = fid_rho
    print(f"fidelity state done.")
    e_states = []  # with c_ops evolution
    # evolving the state under photon loss error
    for count,kap in enumerate(kap_list):
        e_states.append(mesolve(-H,state,t_list,[np.sqrt(kap)*a],[],options=options).states[-1])
        print(f"fidelity {fidelity(norm_fid_rho,e_states[-1])}")
        print(f"e_states {count} done.")
    opList = opListsBs2(the_GKP)
    fidelities,probabilities = [],[]  # initializing lists
    # correcting errors under sBs protocol
    for e_state in e_states:
        rho = e_state/e_state.tr()  # uncorrected state
        prob = 1  # probability of getting that state (neglecting randomness in c_op evolution if there is any) ***
        fidelities.append(fidelity(rho,norm_fid_rho))
        probabilities.append(prob)
        for round in range(max_N_rounds):
            if mode == 'random':
                correction = get_correction(opList,rho)  # get correction at random according to probabilities
            elif mode == 'gg':
                correction = opList[0][0] * opList[1][0]  # get Bgg correction each time
            rho_prime = correction*rho*correction.dag()
            prob_prime = rho_prime.tr()
            rho = rho_prime/prob_prime
            prob *= prob_prime
            if (round+1) % 2:
                rot_rho = Y*rho*Y.dag()
                fidelities.append(fidelity(rot_rho,norm_fid_rho))
            else:
                fidelities.append(fidelity(rho,norm_fid_rho))
            probabilities.append(prob)
    fid_arr,prob_arr = np.real(np.array(fidelities)),np.real(np.array(probabilities))
    # ploting colormaps
    xvec,yvec = rate_list,np.append(N_rounds,max_N_rounds)
    fid = np.reshape(fid_arr, (len(xvec), len(yvec))).T
    pro = np.reshape(prob_arr, (len(xvec), len(yvec))).T
    if mode == 'random':
        fig, ax = plt.subplots(figsize=(10, 8))
        cff = ax.pcolormesh(xvec, yvec, fid,norm=mpl.colors.Normalize(0,1),cmap="seismic")
        fig.colorbar(cff, ax=ax)
        ax.set_title("Fidelity map")
        ax.set_xlabel(r"$\kappa t_{gate}$")
        ax.set_ylabel("N_rounds")
    if mode == 'gg':
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        cff = axs[0].pcolormesh(xvec,yvec,fid,norm=mpl.colors.Normalize(0,1),cmap="seismic")
        cfp = axs[1].pcolormesh(xvec, yvec, pro,norm=mpl.colors.Normalize(0,1),cmap="seismic")
        fig.colorbar(cfp, ax=axs[1])
        axs[0].set_title("Fidelity map")
        axs[1].set_title("Probability map")
        axs[0].set_xlabel(r"$\kappa t_{gate}$")
        axs[1].set_xlabel(r"$\kappa t_{gate}$")
        axs[0].set_ylabel("N_rounds")
        axs[1].sharey=axs[0]
    if save:
        plt.savefig(fig_path+fig_name)
    if show:
        plt.show()
    return


# faire des moyennes des randoms

# *** notion de trajectoire = une mesure d'un phénomène probabiliste
# donc plusieurs trajectoires permettent de calculer une moyenne ***

# faire la moyenne des randoms simplement en additionnant les quatres outcomes possibles

# faire le circuit sBs analytiquement

# voir pourquoi ça devient pas plus pâle

