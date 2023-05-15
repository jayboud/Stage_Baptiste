"""
Author : Jeremie Boudreault
Date: 11/05/2022

Test to see if finite_GKP.py module works
by plotting some Wigner functions.
"""

import numpy as np
from qutip import *
from stage_baptiste.homemades.finite_GKP import get_gkp, GKP
from scipy.constants import pi
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# oscillator

# delta doit être tel que 1/(2*delta^2) ~ <n> ??

delta = 0.3
dim = 50
osc = GKP(delta,dim).state  # gkp with delta = 0
# fig, ax = plot_wigner(osc,method='laguerre')
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"Wigner_test_for_finite_GKP/figs/Wigner_{dim}")


# porte Hadamar

lam = 1e-6
gamma = 0.4
a = destroy(dim)  # produit tensoriel avec qubit
D = displace(dim,0.4)
H = lam*a.dag()*a

tlist = np.linspace(0,pi/(2*lam),1000)

options = Options()
options.nsteps = 10000
res = mesolve(H, osc, tlist, [D],options=options)

# fonction wigner
fig, ax = plot_wigner(res.states[-1])
ax.text(-6,6,rf"$\Delta = {delta}$")
ax.text(-6,5,rf"$N = {dim}$")
plt.savefig(f"Wigner_test_for_finite_GKP/figs/Hadamar_{dim}")

# valeur moyenne operateur deplacement
fig, ax = plt.subplots()
ax.plot(tlist,res.expect[0],label=r"$\langle D(\gamma,t) \rangle$")
ax.text(-6,4,rf"$\gamma = {np.real(gamma)}$")
plt.savefig("Wigner_test_for_finite_GKP/figs/Davg")
