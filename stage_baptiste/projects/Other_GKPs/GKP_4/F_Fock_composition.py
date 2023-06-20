"""
Author : Jeremie Boudreault
Date: 11/05/2022

Code that looks at F kets support in Fock space.
"""

import numpy as np
from qutip import *
from qutip.wigner import _wigner_clenshaw
from stage_baptiste.homemades.finite_GKP import get_d_gkp, GKP
from stage_baptiste.homemades.my_funcs import rot_wigner_clenshaw
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

d = 4
j = 0
delta = 0.3
dim = 100
osc = 1/np.sqrt(2)*(0*GKP(d,0,delta,dim).state + -1*GKP(d,1,delta,dim).state + 0*GKP(d,2,delta,dim).state + 1*GKP(d,3,delta,dim).state)

fig,ax = plt.subplots()
ax.bar(np.arange(0, dim), osc.diag())
plt.show()
