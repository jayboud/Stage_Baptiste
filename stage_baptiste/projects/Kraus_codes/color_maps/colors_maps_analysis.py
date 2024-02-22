"""
Author : Jeremie Boudreault
Date: 4/07/2023
Code that produces color maps of fidelity and probability
for states we want to study under error correction.
"""

import numpy as np
from scipy.constants import pi
from qutip import *
from stage_baptiste.homemades.finite_GKP import GKP, KrausGKP
from stage_baptiste.homemades.KrausOperator_JV import *

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

fig_name = "ks_pios_gg_cmap"
tr_fig_name = "ks_pios_gg_traces"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/"
ground = basis(2,0)
sqrtH = np.cos(np.pi/4)*qeye(2) - 1j*np.sin(np.pi/4)*(sigmax()+sigmaz())/np.sqrt(2)
bqr = sqrtH*ground*ground.dag()*sqrtH.dag()

fid_arr,prob_arr,params = get_fid_n_prob_data(kGKP_obj, H, tgate, max_error_rate, max_N_rounds,
                                              kap_num=10, mode='gg',qubit_mapping=False,pi_o_s=True)
plot_cmaps(fid_arr,prob_arr,*params,
           mode='gg',fig_path=fig_path,pi_o_s=True,
           fig_name=fig_name,save=True,show=False)

plot_fid_traces(fid_arr,*params,traces_ix=[[0,2,4,6,8],[2,4,6,8]],pi_o_s=True,
                fig_path=fig_path,traces_fig_name=tr_fig_name,save=True,show=False)


# methode du paysan (diviser par le max)
# methode du fonctionnaire (améliorer les bins)
# methode du roi (?)
