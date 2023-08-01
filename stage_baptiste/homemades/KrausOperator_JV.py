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
from scipy.optimize import curve_fit
from stage_baptiste.homemades.finite_GKP import GKP
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator


def get_error_states(H,state,dim,t_list,kap_list):
    a = destroy(dim)
    options = Options(nsteps=2000)
    fid_rho = mesolve(-H, state, t_list, [], [], options=options).states[-1]  # reference state for fidelity (no c_ops
    print("fidelity state done.")  # in evolution)
    e_states = []  # with c_ops evolution
    # evolving the state under photon loss error
    for count, kap in enumerate(kap_list):
        e_states.append(mesolve(-H, state, t_list, [np.sqrt(kap)*a], [], options=options).states[-1])
        print(f"fidelity {fidelity(fid_rho, e_states[-1])}")
        print(f"e_states {count} done.")
    return fid_rho, e_states


def opListsBs2(GKP,pi_o_s=False):
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
    if pi_o_s:
        latticeGens = [(np.sqrt(pi*d)+1j*np.sqrt(pi*d)), (-np.sqrt(pi*d)+1j*np.sqrt(pi*d))]
    else:
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


def get_correction(corrections,rho):
    """
    Function that randomly picks an
    index for a list of corrections of len=4.
    Args:
        Klist: list of Qobj
            List of operators used to apply sBs error correction.
        rho: Qobj (operator)
            The density matrix to correct.

    Returns: int
        The index.

    """
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
        get_correction(corrections,rho)
    return correct_with


def qb_mapping_fidelity(state,state_ref,basic_qubit_ref):
    """
    Function that calculates fidelity according to
    a qubit mapping. (See mapping notes.pdf)
    Args:
        state: Qobj (density matrix)
            The GKP state to map
        state_ref: Qobj (ket)
            The reference GKP state to map
        basic_qubit_ref:
            The basic qubit density matrix to reference fidelity (no mapping involved).

    Returns:
        fidelity: (float)
            The fidelity according to the mapping.

    """
    dim = state.shape[0]
    Ds = np.sqrt(pi/2), np.sqrt(pi/2)*(1+1j), np.sqrt(pi/2)*1j
    # --------- bin method --------------
    # a = destroy(dim)
    # x_op = (a + a.dag())/np.sqrt(2)
    # p_op = (a - a.dag())/(1j*np.sqrt(2))
    # eig_vals, bins, eig_states = [], [], []
    # for op in [x_op, p_op]:
    #     op_eigvals,op_eig_states = op.eigenstates()
    #     eig_vals.append(op_eigvals)
    #     eig_states.append(op_eig_states)
    #     op_bins = np.copy(op_eigvals)
    #     condition = abs(abs(op_bins)/np.sqrt(pi) - abs(op_bins)/np.sqrt(pi)//1)
    #     up_bins, down_bins = condition < 1/2, condition > 1/2
    #     op_bins[up_bins], op_bins[down_bins] = 1, -1
    #     bins.append(op_bins)
    # Z = Qobj(np.sum([a_bin*eig_state*eig_state.dag() for a_bin,eig_state in zip(bins[0],eig_states[0])],axis=0))
    # X = Qobj(np.sum([a_bin*eig_state*eig_state.dag() for a_bin,eig_state in zip(bins[1],eig_states[1])],axis=0))
    # Y = -1j*Z*X
    # ---------------- other method ----------------
    X,Y,Z = [displace(dim, gamma) for gamma in Ds]  # X,Z,Y
    r_qubit,r_ref = [np.array([np.real((op*st).tr()) for op in [X,Y,Z]]) for st in [state,state_ref]]
    r_qubit,r_ref = [np.array([np.real((op*st).tr()) for op in [X,Y,Z]]) for st in [state,state_ref]]
    sigs = np.array([sigmax(),sigmay(),sigmaz()])
    mapped_qubit,mapped_qubit_ref = [(qeye(2) + Qobj(sum(r[:,None,None]*sigs)))/2 for r in [r_qubit,r_ref]]
    # fid = fidelity(mapped_qubit,basic_qubit_ref)  # with bqr
    fid = fidelity(mapped_qubit,mapped_qubit_ref)  # both mapped
    return fid


def get_fidelities(A,B,basic_qubit_ref):
    """
    Function that get normal and qubit mapping fidelities.
    Args:
        A: Qobj (matrix)
        B: Qobj (matrix)
        basic_qubit_ref: Qobj (matrix)

    Returns:
        The fidelities.

    """
    if basic_qubit_ref:
        out = [fidelity(A,B),qb_mapping_fidelity(A,B,basic_qubit_ref)]
    else:
        out = [fidelity(A,B)]
    return out


def get_fid_n_prob_data(GKP_obj,H,t_gate,max_error_rate,max_N_rounds,t_num=10,kap_num=10,N_rounds_steps=1,mode='random',
               qubit_mapping=False,bqr=None,superposition_state=None,ss_d=None,ss_delta=None,ss_hilbert_dim=None,pi_o_s=False):
    """
    Function that calculates fidelity colormap data (and probability if mode=gg) of sBs error correction
    protocol applied to a qubit state evolving under photon loss error sqrt(kappa)*a .
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
            'Avg' traces the average fidelity map of all corrections weighted by their probability.
        qubit_mapping: bool
            Decides wether or not to use qubit mapping
            to calculate fidelity. (see mapping notes.pdf)
        bqr: Qobj (matrix)
            The basic qubit reference state for the fidelity of mapped qubits.
        superposition_state: Qobj (ket)
            The initial state as a superposition of many GKP.states .
        ss_d: int (if superposition_state)
            The number d of logical states of the qudit space
            of states in the superposition.
        ss_delta: float (if superposition)
            The enveloppe on the finite stats of the superposition.
        ss_hilbert_dim: int (if superposition_state)
            The hilbert dimension in Fock space of states in the superposition state.
    Returns:
        fid_arr: ndarray of floats
            Fidelity data.
        prob_arr: ndarray of floats
            Probability data.
        params: list [ndarray of floats, ndarray of ints, int]
            [rate_list, N_rounds, max_N_rounds].

    """
    if mode not in ['random','gg','avg']:
        raise ValueError("You have to choose a mode between 'random' or 'gg' or 'avg.")
    if qubit_mapping:
        if not isinstance(bqr,Qobj):
            raise ValueError("You have to define a valid basic qubit reference state to calculate"
                             "the fidelity of mapped qubits with kwarg 'bqr'.")
    # dealing with a simple GKP state or a superposition of states.
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
    fid_rho,e_states = get_error_states(H,state,dim,t_list,kap_list)
    opList = opListsBs2(the_GKP,pi_o_s=pi_o_s)
    corrections = [opList[0][0] * opList[1][0], opList[0][0] * opList[1][1],  # [Bgg, Bge
                   opList[0][1] * opList[1][0], opList[0][1] * opList[1][1]]  # Beg, Bee]
    fidelities,probabilities = [],[]  # initializing lists
    # correcting errors under sBs protocol
    for e_state in e_states:
        rho = e_state/e_state.tr()  # uncorrected state
        prob = 1  # probability of getting that state (neglecting randomness in c_op evolution if there is any) ***
        fidelities.append(get_fidelities(rho,fid_rho,bqr)[qubit_mapping])
        probabilities.append(prob)
        for n_round in range(max_N_rounds):
            if mode == 'random':
                correction = get_correction(corrections,rho)  # get correction at random according to probabilities
                rho_prime = correction*rho*correction.dag()
            elif mode == 'gg':
                correction = opList[0][0]*opList[1][0]  # get Bgg correction each time
                rho_prime = correction*rho*correction.dag()
            elif mode == 'avg':
                rho_prime = sum([correction*rho*correction.dag() for correction in corrections])
            prob_prime = rho_prime.tr()
            rho = rho_prime/prob_prime
            prob *= prob_prime
            if (n_round+1) % 2:
                rot_rho = Y*rho*Y.dag()
                fidelities.append(get_fidelities(rot_rho,fid_rho,bqr)[qubit_mapping])
            else:
                fidelities.append(get_fidelities(rho,fid_rho,bqr)[qubit_mapping])
            probabilities.append(prob)
    fid_arr,prob_arr = np.real(np.array(fidelities)),np.real(np.array(probabilities))
    params = [rate_list,N_rounds,max_N_rounds]
    return fid_arr,prob_arr,params


def plot_cmaps(fid_arr,prob_arr,*params,mode='gg',fig_path=None,fig_name=None,save=True,show=False):
    """
    Function that plots a fidelity colormap (and a probability colormap if mode=gg). The x and y axis
    of the colormap are respectively the dimensionless error rate kappa*tgate (kappa is in tgate units)
    and the number of error correcting rounds.
    Args:
        fid_arr: ndarray of floats
            Fidelity data.
        prob_arr: ndarray of floats
            Probability data.
        *params: [ndarray of floats, ndarray of ints, int]
            [rate_list,N_rounds,max_N_rounds].
        mode:
        fig_name:
        fig_path: string
            The path to save the figure.
        save: bool
            Wether to save or not the figure at fig_path.
        show: bool
            Wether to show or not the figure.

    Returns:
        A colormap of fidelity (and probability if 'gg').

    """
    rate_list, N_rounds, max_N_rounds = params
    xvec,yvec = rate_list,np.append(N_rounds,max_N_rounds)
    fid = np.reshape(fid_arr, (len(xvec), len(yvec))).T
    pro = np.reshape(prob_arr, (len(xvec), len(yvec))).T
    if mode in ['random','avg']:
        fig, ax = plt.subplots(figsize=(10, 8))
        cff = ax.pcolormesh(xvec, yvec, fid,norm=mpl.colors.Normalize(0,1),cmap="seismic")
        fig.colorbar(cff, ax=ax)
        ax.set_title("Fidelity map")
        ax.set_xlabel(r"$\kappa t_{gate}$")
        ax.set_ylabel("N_rounds")
    elif mode == 'gg':
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


def plot_fid_traces(fid_arr,*params,traces_ix=[[8,10,12,14],[2,4,6,8]],fig_path=None,traces_fig_name=None,save=True,show=False):
    """
    Function that plots fidelity traces for
    chosen error rate and/or number of error correcting rounds.

    Args:
        fid_arr: ndarray of floats
            Fidelity data.
        *params: [ndarray of floats, ndarray of ints, int]
            [rate_list,N_rounds,max_N_rounds].
        traces_ix: list of ints
            The indexes that select error rates and/or
            number of error correcting rounds.
        fig_path: string
            The path to save the figure.
        traces_fig_name: string
            The name of the figure.
        save: bool
            Wether to save or not the figure at fig_path.
        show: bool
            Wether to show or not the figure.

    Returns:
        A plot of fidelity traces.

    """
    rate_list, N_rounds, max_N_rounds = params
    xvec, yvec = rate_list, np.append(N_rounds, max_N_rounds)
    fid = np.reshape(fid_arr, (len(xvec), len(yvec))).T
    # horizontal and vertical traces
    if not isinstance(traces_ix[0],list):
        traces_ix = [traces_ix,[]]
    if not traces_ix[0] and not traces_ix[1]:
        raise ValueError("You have to choose some indexes to plot in"
                         "traces_ix = [[n_rounds_ixs],[error_rates_ixs]]. "
                         "Providing one list will be assumed to be n_rounds_idx.")
    for i in [0,1]:
        if not traces_ix[i]:
            traces_ix[i] = traces_ix[i-1]  # makes sure that both v_ixs and h_ixs exist
    horizontals = np.array(fid[traces_ix[0],:])
    # horizontals = np.array(fid[traces_ix[0],:])/fid[8,0]  # normalisation
    verticals = np.array(fid[:, traces_ix[1]])
    # verticals = np.array(fid[:, traces_ix[1]])/fid[8,0]  # normalisation
    def parabolic(x, a, b, c):
        return a*x**2 + b*x + c
    fig,axs = plt.subplots(1,2, figsize=(10,4))
    h_traces = axs[0].plot(xvec,horizontals.T)
    v_traces = axs[1].plot(yvec,verticals)
    for h_trace,h_ix in zip(h_traces,traces_ix[0]):
        x_data,y_data = h_trace.get_data()[0],h_trace.get_data()[1]
        popt,pcov = curve_fit(parabolic,x_data,y_data)
        x_fit = np.linspace(x_data[1],max(h_trace.get_data()[0]),200)
        y_fit = parabolic(x_fit,*popt)
        axs[0].plot(x_fit,y_fit,label=rf"fit $N={h_ix},\alpha={round(popt[0],3)},\beta = {round(popt[1],3)}, \gamma= {round(popt[2],3)}$",ls="dotted",color=h_trace.get_color())
    axs[0].text(0.05,min(y_fit),r"$F = \alpha x^2 + \beta x + \gamma$")
    for v_trace,v_ix in zip(v_traces,traces_ix[1]):
        v_trace.set(label=r"$\kappa t_{gate} = $"+f"{round(xvec[v_ix],3)}")
    axs[0].set_title("Horizontal traces")
    axs[1].set_title("Vertical traces")
    axs[0].set_xlabel(r"$\kappa t_{gate}$")
    axs[1].set_xlabel("N_rounds")
    axs[0].set_ylabel(r"$F$",rotation=0)
    # axs[0].set_ylim(0,1)
    axs[1].sharey = axs[0]
    for ax in axs:
        ax.legend(loc="upper right")
    plt.legend()
    if save:
        plt.savefig(fig_path + traces_fig_name)
    if show:
        plt.show()

    return
