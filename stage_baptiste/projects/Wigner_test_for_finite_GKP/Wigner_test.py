"""
Author : Jeremie Boudreault
Date: 11/05/2022

Test to see if finite_GKP.py module works
by plotting some Wigner functions.
"""

import numpy as np
import qutip as qt
from stage_baptiste.homemades.finite_GKP import get_gkp, GKP
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.3
dim = 75
gkp = GKP(delta,dim)  # gkp with delta = 0
fig, ax = qt.plot_wigner(gkp.state,method='laguerre',alpha_max=1)
plt.savefig(f"Wigner_test_for_finite_GKP/figs/Wigner_{dim}")
