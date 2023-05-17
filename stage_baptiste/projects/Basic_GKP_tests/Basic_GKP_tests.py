"""
Author : Jeremie Boudreault
Date: 11/05/2022

Test to see if finite_GKP.py module works.
"""

import numpy as np
from qutip import *
from stage_baptiste.homemades.finite_GKP import get_gkp, GKP
from scipy.constants import pi
from matplotlib import pyplot as plt

# oscillator

# delta doit être tel que 1/(2*delta^2) ~ <n> ??

delta = 0.25
dim = 100
osc = GKP(delta,dim).state  # gkp with delta = 0
# fig, ax = plot_wigner(osc,method='laguerre')
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"Wigner_test_for_finite_GKP/figs/Wigner_{dim}")


# porte Hadamar


a = destroy(dim)  # produit tensoriel avec qubit
X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
Ds_labels = ["X","Y","Z","Sx","Sp"]
H = a.dag()*a

tlist = np.linspace(0,pi/2,1000)
outs = [mesolve(H, osc, tlist, [], [D]) for D in Ds]

# fonction wigner
# fig, ax = plot_wigner(res.states[-1])
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"Wigner_test_for_finite_GKP/figs/Hadamar_{dim}")

# valeur moyenne operateur deplacement
fig, ax = plt.subplots()
for out,label in zip(outs,Ds_labels):
    line_re = ax.plot(tlist,np.real(out.expect[0]),label=rf"${label}$")  # real part
    line_im = ax.plot(tlist,np.imag(out.expect[0]),ls="--",color=line_re[0].get_color())  # imaginary part
ax.text(0.02,0.8,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
ax.text(0.02,0.7,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
ax.text(0.02,0.6,r"$H = a^{\dag}a$",ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle D \rangle$",rotation=0)
plt.title("Valeur moyenne du déplacement en fonction du temps")
plt.legend()
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/H_Davg_{dim}")


# porte racine Hadamard
n = a.dag()*a
H = a.dag()*a*a.dag()*a - 4*a.dag()*a

tlist = np.linspace(0,3*pi/8,1000)
outs = [mesolve(H, osc, tlist, [], [D]) for D in Ds]

# fonction wigner
# outa = mesolve(H, osc, tlist, [], [])  # car avec les e_ops je ne trouve pas les etats etrangement
# fig, ax = plot_wigner(outa.states[-1])
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/sqrtHadamar_{dim}")


fig, ax = plt.subplots()
for out,label in zip(outs,Ds_labels):
    line_re = ax.plot(tlist,np.real(out.expect[0]),label=rf"${label}$")  # real part
    line_im = ax.plot(tlist,np.imag(out.expect[0]),ls="--",color=line_re[0].get_color())  # imaginary part
ax.text(0.02,0.8,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
ax.text(0.02,0.7,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
ax.text(0.02,0.6,r"$H = n^2 - 4n$",ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\langle D \rangle$",rotation=0)
plt.title("Valeur moyenne du déplacement en fonction du temps")
plt.legend()
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/sqrtH_Davg_{dim}")
