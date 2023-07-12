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
max_error_rate = 0.20
max_N_rounds = 30
fig_name = "sqrtH_gg_cmap"
fig_path = f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/color_maps/figs/"
color_maps(GKP_obj, H, tgate, max_error_rate, max_N_rounds, kap_num=10, mode='gg',fig_name=fig_name, fig_path=fig_path)
