"""
Author : Jeremie Boudreault
Date: 11/05/2022

Code that looks at F kets support in Fock space.
"""

import numpy as np
from numpy import sqrt
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
F1 = 1/np.sqrt(6)*(2*GKP(d,0,delta,dim).state+
                   1*GKP(d,1,delta,dim).state +
                   0*GKP(d,2,delta,dim).state +
                   1*GKP(d,3,delta,dim).state).unit()
F2 = 1/np.sqrt(2)*(1*GKP(d,0,delta,dim).state+
                   0*GKP(d,1,delta,dim).state +
                   1*GKP(d,2,delta,dim).state +
                   0*GKP(d,3,delta,dim).state)
F3 = 1/2*(-1*GKP(d,0,delta,dim).state+
                   1*GKP(d,1,delta,dim).state +
                   1*GKP(d,2,delta,dim).state +
                   1*GKP(d,3,delta,dim).state)
F4 = 1/np.sqrt(2)*(0*GKP(d,0,delta,dim).state+
                   -1*GKP(d,1,delta,dim).state +
                   0*GKP(d,2,delta,dim).state +
                   1*GKP(d,3,delta,dim).state)

# le delta n'affecte pas tous les états de la même façon
# vu le différent support de chacun dans l'espace de Fock
i = 3
if i == 1:
    osc = F1
elif i == 2:
    osc = F2
elif i == 3:
    osc = F3
elif i == 4:
    osc = F4
a = destroy(dim)
n_op = a.dag()*a
H = -n_op**2
tlist = np.linspace(0,pi/16,200)
options = Options(store_states=True)  # get states even if e_ops are calculated
outs = mesolve(H, osc, tlist, [], [],options=options)
coeffs = np.squeeze(outs.states[0][:,0])
f_coeffs = np.squeeze(outs.states[-1][:,0])


fig,ax = plt.subplots()
ax.bar(np.arange(0, dim), np.real(coeffs))
ax.bar(np.arange(0, dim)+1, np.imag(coeffs))
fig.suptitle(rf"Fock composition of $ |F_{i}\rangle$")
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/F_tests/figs/F_{i}_fock_comp")

fig,ax = plt.subplots()
ax.bar(np.arange(0, dim), np.real(np.exp(-1j*pi/4)*f_coeffs))
ax.bar(np.arange(0, dim)+1, np.imag(np.exp(-1j*pi/4)*f_coeffs))
ax.text(4,0.6,r"$U = e^{i\frac{\pi}{16}n^2}$")
ax.text(4,0.5,r"$e^{i\frac{\pi}{4}}U|F_3\rangle$")
fig.suptitle(rf"Fock composition of $ |F_{i}\rangle$")
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/F_tests/figs/F_{i}_fock_comp_final")


fig, ax = plot_wigner(outs.states[0])
ax.text(-6,6,rf"$\Delta = {delta}$")
ax.text(-6,5,rf"$N = {dim}$")
ax.text(-6,4,rf"$|\psi\rangle = |F_{i}\rangle$")
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/F_tests/figs/F_{i}_state")

fig, ax = plot_wigner(outs.states[-1])
ax.text(-6,7,rf"$\Delta = {delta}$")
ax.text(-6,6,rf"$N = {dim}$")
ax.text(-6,5,r"$U = e^{i\frac{\pi}{16}n^2}$")
ax.text(-6,4,rf"$|\psi\rangle = |F_{i}\rangle$")
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/F_tests/figs/F_{i}_state_final")
