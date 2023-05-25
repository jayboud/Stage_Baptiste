"""
Author : Jeremie Boudreault
Date: 11/05/2022

Tests to see if finite_GKP.py module works properly.
"""

import numpy as np
from qutip import *
from stage_baptiste.homemades.finite_GKP import get_d_gkp, GKP
from scipy.constants import pi
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# oscillator

# delta has to be 1/(2*delta^2) ~ <n> ??

delta = 0.25
dim = 100
osc = GKP(delta,dim,2,0).state  # gkp with delta = 0
# fig, ax = plot_wigner(osc,method='laguerre')
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"Wigner_test_for_finite_GKP/figs/Wigner_{dim}")


# Hadamar gate (e^{i pi/4 a.dag a})


a = destroy(dim)
X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
Ds_labels = ["X","Y","Z","Sx","Sp"]
H = a.dag()*a


# wigner function of d_GKP and some gate on it
m = 4
k = 0
_,_,d_osc = get_d_gkp(delta,dim,m,k)  # oscillator with d states
fig, ax = plot_wigner(d_osc)
ax.text(-6,6,rf"$\Delta = {delta}$")
ax.text(-6,5,rf"$N = {dim}$")
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/GKP_{dim}_m={m},k={k}")
n = a.dag()*a
H = n**2
tlist = np.linspace(0,pi/16,10)
options = Options(store_states=True)  # get states even if e_ops are calculated
out = mesolve(H, d_osc, tlist, [], [])
fig, ax = plot_wigner(out.states[-1])
ax.text(-6,6.5,rf"$\Delta = {delta}$")
ax.text(-6,5.8,rf"$N = {dim}$")
ax.text(-6,5.0,r"$U = e^{i\frac{\pi}{16}n^2}$")
# mesuring dimension of grid
ax.plot([0,np.sqrt(pi/2)],[0,np.sqrt(pi/2)],'-',lw=1.5,color="black")
ax.text(np.sqrt(pi/2)/2,np.sqrt(pi/2)/2-0.5,r"$\sqrt{\pi}$",color="black",rotation=45)

plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/somegate_GKP_{dim}_m={m},k={k}")


# average of displacements for H
# fig, ax = plt.subplots()
# for out,label in zip(outs,Ds_labels):
#     line_re = ax.plot(tlist,np.real(out.expect[0]),label=rf"${label}$")  # real part
#     line_im = ax.plot(tlist,np.imag(out.expect[0]),ls="--",color=line_re[0].get_color())  # imaginary part
# ax.text(0.02,0.8,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
# ax.text(0.02,0.7,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
# ax.text(0.02,0.6,r"$H = a^{\dag}a$",ha='left', va='top', transform=ax.transAxes)
# ax.set_xlabel(r"$t$")
# ax.set_ylabel(r"$\langle D \rangle$",rotation=0)
# plt.title("Valeur moyenne du déplacement en fonction du temps")
# plt.legend()
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/H_Davg_{dim}")


# sqrtH gate


# n = a.dag()*a
# H = n*n
#
# tf = pi/8
# # tlist = np.linspace(0,tf,150)  # times for animation
# tlist = np.linspace(0,tf,1200)  # times for expectation values
# options = Options(store_states=True)  # get states even if e_ops are calculated
# outs = [mesolve(H, osc, tlist, [], [D],options=options) for D in Ds]


# sqrtH Wigner function


# fig, ax = plot_wigner(outs[0].states[-1])
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/sqrtHadamar_{dim}")


# sqrtH Wigner function animation


# debut = np.zeros(4,dtype=int).tolist()
# fin = (tf*np.ones(4)).tolist()
# gif_tlist = debut+tlist.tolist()+fin  # times (0 to 3pi/8) with supplementary 0s and 3pi/8s for gif
# frames = debut+np.arange(0,len(tlist),1).tolist()+(range(len(tlist))[-1]*np.ones(4,dtype=int)).tolist()  # index (0,1,2,3,...)
# fig, ax = plt.subplots(figsize=(5, 4))
# ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
# ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
# ax.set_title("Wigner function", fontsize=12)
# xvec = np.linspace(-7.5, 7.5, 200)
# W0 = wigner(outs[0].states[0], xvec, xvec)
# W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
# wlim = abs(W).max()
# cax = ax.contourf(xvec, yvec, W, 100,norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
#
#
# def animate(i):
#     ax.clear()
#     ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
#     ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
#     ax.set_title("Wigner function", fontsize=12)
#     W0 = wigner(outs[0].states[i], xvec, xvec)
#     W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
#     ax.contourf(xvec, yvec, W, 100, norm=mpl.colors.Normalize(-wlim, wlim), cmap=mpl.colormaps['RdBu'])
#     ax.text(0.02,0.75,rf"$t = {round(gif_tlist[i],4)}$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.8,r"$H = n^2$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.85,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.9,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
#
#
# anim = FuncAnimation(
#     fig, animate, interval=100, frames=frames)
# anim.save("/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/"
#           "Wigner_animation_sqrtH.gif")


# average of displacements for sqrtH

# fig, ax = plt.subplots()
# for out,label in zip(outs,Ds_labels):
#     line_re = ax.plot(tlist,np.real(out.expect[0]),label=rf"${label}$")  # real part
#     line_im = ax.plot(tlist,np.imag(out.expect[0]),ls="--",color=line_re[0].get_color())  # imaginary part
# ax.text(0.02,0.9,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
# ax.text(0.02,0.85,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
# ax.text(0.02,0.8,r"$H = n^2$",ha='left', va='top', transform=ax.transAxes)
# ax.set_xlabel(r"$t$")
# ax.set_ylabel(r"$\langle D \rangle$",rotation=0)
# plt.title("Valeur moyenne du déplacement en fonction du temps")
# plt.legend()
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_tests/figs/sqrtH_Davg_{dim}")
