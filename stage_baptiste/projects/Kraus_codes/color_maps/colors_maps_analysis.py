"""
Author : Jeremie Boudreault
Date: 4/07/2023
Code that produces color maps of fidelity and probability
for states we want to study under error correction.
"""

import numpy as np
from numpy import sqrt
from scipy.constants import pi
from qutip import *
from stage_baptiste.homemades.finite_GKP import GKP, KrausGKP
from stage_baptiste.homemades.KrausOperator_JV import *

"""# one error correcting procedure

d = 2
j = 0
delta = 0.15
hilbert_dim = 75
a = destroy(hilbert_dim)
N_op = a.dag()*a

kGKP_obj = KrausGKP(d,j,delta,hilbert_dim)
H = N_op**2
tgate = pi/16
max_error_rate = 0.01
max_N_rounds = 30

fig_name = "ks_pios_gg_cmap_perfect_ref_mapped"
tr_fig_name = "ks_pios_gg_traces_perfect_ref_mapped"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/direct_gate/"
ground = basis(2,0)
sqrtH = np.cos(np.pi/4)*qeye(2) - 1j*np.sin(np.pi/4)*(sigmax()+sigmaz())/np.sqrt(2)
bqr = sqrtH*ground*ground.dag()*sqrtH.dag()
# Special middle state at pi/16
exp = np.exp(1j*pi/4)
# alpha_0 = sqrt(6)/2*(1 + exp)
# beta_0 = -sqrt(2)/2*(1 + exp)
# gamma_0 = (-1 + exp)*exp
alpha_0 = sqrt(6)/4*(1 + exp)
beta_0 = -sqrt(2)/4*(1 + exp)
gamma_0 = (-1 + exp)*exp/2  # le *exp change tout!
c0 = 2*alpha_0/sqrt(6) + beta_0/sqrt(2) - gamma_0/2
c1 = alpha_0/sqrt(6) + gamma_0/2
c2 = beta_0/sqrt(2) + gamma_0/2
c3 = alpha_0/sqrt(6) + gamma_0/2
perf_pi_o_s = 1/2*(c0*GKP(d,0,delta,hilbert_dim).state+c1*GKP(d,1,delta,hilbert_dim).state+c2*GKP(d,2,delta,hilbert_dim).state+c3*GKP(d,3,delta,hilbert_dim).state)
rho_ref = ket2dm(perf_pi_o_s)

# bqr in not useful in the calculation
fid_arr,prob_arr,last_state,params = get_fid_n_prob_data(kGKP_obj, H, tgate, max_error_rate, max_N_rounds,
                                              kap_num=10, mode='gg',reference_state=rho_ref,qubit_mapping=True,bqr=bqr,pi_o_s=True)
plot_cmaps(fid_arr,prob_arr,*params,
           mode='gg',fig_path=fig_path,halfs_only=True,
           fig_name=fig_name,save=True,show=False)

plot_fid_traces(fid_arr,*params,traces_ix=[[0,2,4,6,8],[2,4,6,8]],halfs_only=True,
                fig_path=fig_path,traces_fig_name=tr_fig_name,save=True,show=False)

"""

# error correcting procedure in two steps
d = 2
j = 0
delta = 0.15
hilbert_dim = 75
a = destroy(hilbert_dim)
N_op = a.dag()*a

kGKP_obj = KrausGKP(d,j,delta,hilbert_dim)
H = N_op**2
tgate = pi/16
max_error_rate = 0.01
max_N_rounds = 30
kap_num = 10
kap_max = max_error_rate/tgate
kap_list = np.linspace(0,kap_max,kap_num)
rate_list = np.linspace(0,max_error_rate,kap_num)
N_rounds = np.arange(0,max_N_rounds,1)

fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/two_half_gate/"
ground = basis(2,0)
sqrtH = np.cos(np.pi/4)*qeye(2) - 1j*np.sin(np.pi/4)*(sigmax()+sigmaz())/np.sqrt(2)
bqr = sqrtH*ground*ground.dag()*sqrtH.dag()
# Special middle state at pi/16
exp = np.exp(1j*pi/4)
# alpha_0 = sqrt(6)/2*(1 + exp)
# beta_0 = -sqrt(2)/2*(1 + exp)
# gamma_0 = (-1 + exp)*exp
# alpha_0 = sqrt(6)/4*(1 + exp)
# beta_0 = -sqrt(2)/4*(1 + exp)
# gamma_0 = (-1 + exp)*exp/2  # le *exp change tout!
# c0 = 2*alpha_0/sqrt(6) + beta_0/sqrt(2) - gamma_0/2
# c1 = alpha_0/sqrt(6) + gamma_0/2
# c2 = beta_0/sqrt(2) + gamma_0/2
# c3 = alpha_0/sqrt(6) + gamma_0/2
# perf_pi_o_s = 1/2*(c0*GKP(d,0,delta,hilbert_dim).state+c1*GKP(d,1,delta,hilbert_dim).state+c2*GKP(d,2,delta,hilbert_dim).state+c3*GKP(d,3,delta,hilbert_dim).state)
# rho_ref = ket2dm(perf_pi_o_s)
options = Options(nsteps=2000)
rho_ref = mesolve(-H, kGKP_obj.state, np.linspace(0, pi/8, 10), [], [], options=options).states[-1]

# bqr and rho_ref
# not useful in the calculation
middle_rounds = 8
fig_name = f"split_{middle_rounds}mr_ks_gg_cmap_mapped"
tr_fig_name = f"split_{middle_rounds}mr_ks_gg_traces_mapped"
fid_arr,prob_arr,final_states,params = get_fid_n_prob_data(kGKP_obj, H, tgate, max_error_rate, middle_rounds,
                                              kap_num=10, mode='gg',reference_state=rho_ref,qubit_mapping=False,bqr=bqr,pi_o_s=True)
N_middle_rounds = np.arange(0,middle_rounds,1)
xvec,yvec = xvec, yvec = rate_list, N_middle_rounds[:middle_rounds//2+1]
fid_arr = np.reshape(fid_arr, (len(xvec), len(yvec))).T
# -------- second half --------- #
opList = opListsBs2(kGKP_obj,pi_o_s=False)
corrections = [opList[0][0] * opList[1][0], opList[0][0] * opList[1][1],  # [Bgg, Bge
                   opList[0][1] * opList[1][0], opList[0][1] * opList[1][1]]  # Beg, Bee]
fidelities,probabilities = [],[]  # initializing lists
mode = 'gg'
for i,(final_state,kappa) in enumerate(zip(final_states,kap_list)):
    qubit_mapping = False
    prob = fid_arr[:,i][-1]  # probability for final state
    rho = final_state/final_state.tr()
    fidelities.append(1 - get_fidelities(rho, rho_ref, bqr)[qubit_mapping])
    probabilities.append(prob)
    t_num = 10
    t_list = np.linspace(0, tgate, t_num)
    final_ev = mesolve(-H, rho, t_list, [np.sqrt(kappa)*a], [], options=options).states[-1]
    # apply corrections
    for i, n_round in enumerate(range(max_N_rounds)):
        if mode == 'random':
            correction = get_correction(corrections, rho)  # get correction at random according to probabilities
            rho_prime = correction * rho * correction.dag()
        elif mode == 'gg':
            correction = opList[0][0] * opList[1][0]  # get Bgg correction each time
            rho_prime = correction * rho * correction.dag()
        elif mode == 'avg':
            rho_prime = sum([correction * rho * correction.dag() for correction in corrections])
        prob_prime = rho_prime.tr()
        rho = rho_prime / prob_prime
        prob *= prob_prime
        if (n_round+1) % 2:  # car round 0 déjà faite donc décalage de 1 pour tous
            pass
        else:  # bonnes rounds (paires)
            fidelities.append(1 - get_fidelities(rho, rho_ref, bqr)[qubit_mapping])
            probabilities.append(prob)
fid_arr2, prob_arr2 = np.real(np.array(fidelities)), np.real(np.array(probabilities))
params2 = [rate_list, N_rounds, max_N_rounds]


plot_cmaps(fid_arr2,prob_arr2,*params2,
           mode='gg',fig_path=fig_path,halfs_only=True,
           fig_name=fig_name,save=True,show=False)

plot_fid_traces(fid_arr2,*params2,traces_ix=[[0,2,4,6,8],[2,4,6,8]],halfs_only=True,
                fig_path=fig_path,traces_fig_name=tr_fig_name,save=True,show=False)
