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

# one error correcting procedure
"""
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

fig_name = "ks_pios_gg_cmap_perfect_ref_notmapped"
tr_fig_name = "ks_pios_gg_traces_perfect_ref_notmapped"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/direct_gate"
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
fid_arr,prob_arr,params,last_state = get_fid_n_prob_data(kGKP_obj, H, tgate, max_error_rate, max_N_rounds,
                                              kap_num=10, mode='gg',reference_state=rho_ref,qubit_mapping=False,bqr=bqr,pi_o_s=True)
plot_cmaps(fid_arr,prob_arr,*params,
           mode='gg',fig_path=fig_path,pi_o_s=True,
           fig_name=fig_name,save=True,show=False)

plot_fid_traces(fid_arr,*params,traces_ix=[[0,2,4,6,8],[2,4,6,8]],pi_o_s=True,
                fig_path=fig_path,traces_fig_name=tr_fig_name,save=True,show=False)
"""

# error correcting procdure in two steps
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

fig_name = "ks_pios_gg_cmap_perfect_ref_notmapped"
tr_fig_name = "ks_pios_gg_traces_perfect_ref_notmapped"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/two_half_gate"
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
middle_rounds = 5
fid_arr,prob_arr,final_states,params = get_fid_n_prob_data(kGKP_obj, H, tgate, max_error_rate, middle_rounds,
                                              kap_num=10, mode='gg',reference_state=rho_ref,qubit_mapping=False,bqr=bqr,pi_o_s=True)

for final_state, kappa in zip(final_states,kap_list):
    a = 1

plot_cmaps(fid_arr,prob_arr,*params,
           mode='gg',fig_path=fig_path,pi_o_s=True,
           fig_name=fig_name,save=True,show=False)

plot_fid_traces(fid_arr,*params,traces_ix=[[0,2,4,6,8],[2,4,6,8]],pi_o_s=True,
                fig_path=fig_path,traces_fig_name=tr_fig_name,save=True,show=False)

