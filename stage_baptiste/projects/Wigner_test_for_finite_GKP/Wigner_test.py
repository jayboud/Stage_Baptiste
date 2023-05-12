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

# delta doit être tel que 1/(2*delta^2) ~ <n>

delta = 0.1
dim = 120
gkp = GKP(delta,dim)  # gkp with delta = 0
fig, ax = qt.plot_wigner(gkp.state,method='laguerre')
plt.savefig(f"Wigner_test_for_finite_GKP/figs/Wigner_{dim}")
