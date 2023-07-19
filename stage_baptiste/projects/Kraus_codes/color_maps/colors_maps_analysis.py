"""
Author : Jeremie Boudreault
Date: 4/07/2023
Code that produces color maps of fidelity and probability
for states we want to study under error correction.
"""

import numpy as np
from scipy.constants import pi
from qutip import *
from stage_baptiste.homemades.finite_GKP import GKP
from stage_baptiste.homemades.KrausOperator_JV import color_maps

d = 2
j = 0
delta = 0.15
hilbert_dim = 75
a = destroy(hilbert_dim)
N_op = a.dag()*a

GKP_obj = GKP(d,j,delta,hilbert_dim)
H = N_op**2
tgate = pi/8
max_error_rate = 1.0
max_N_rounds = 30
fig_name = "qbmapping_sqrtH_gg_cmap_2"
tr_fig_name = "qbmapping_sqrtH_quad_traces_2"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/"
ground = basis(2,0)
sqrtH = np.cos(np.pi/4)*qeye(2) - 1j*np.sin(np.pi/4)*(sigmax()+sigmaz())/np.sqrt(2)
bqr = sqrtH*ground*ground.dag()*sqrtH.dag()
color_maps(GKP_obj, H, tgate, max_error_rate, max_N_rounds, kap_num=15, mode='gg',
           traces=True,traces_ix=[[2,4,6,8],[2,4,6,8]],qubit_mapping=True,bqr=bqr,
           fig_name=fig_name,traces_fig_name=tr_fig_name,fig_path=fig_path,show=True)


# methode du paysan (diviser par le max)
# methode du fonctionnaire (améliorer les bins)
# methode du roi
