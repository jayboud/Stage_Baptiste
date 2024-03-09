"""
Author : Jeremie Boudreault
Date: 11/05/2023

Making various basic calculation using finite_GKP.py .
"""

import numpy as np

from qutip import *
from stage_baptiste.homemades.finite_GKP import GKP, KrausGKP
from stage_baptiste.homemades.general_funcs import *
from scipy.constants import pi
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# oscillator

# delta has to be 1/(2*delta^2) ~ <n> ??

delta = 0.25
dim = 100
# osc = GKP(2,0,delta,dim).state  # gkp with delta = 0.25
# # random = (basis(20,4)+basis(20,3)+basis(20,9))/np.sqrt(3)
# plus = 1/np.sqrt(2)*(GKP(2,0,delta,dim).state+GKP(2,1,delta,dim).state)
# fig, ax = plot_wigner(osc,method='laguerre')
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# ax.plot([0,0],[0,2*np.sqrt(pi)],'-',lw=1.5,color="black")
# ax.plot([0,2*np.sqrt(pi)],[0,0],'-',lw=1.5,color="black")
# ax.text(-np.sqrt(pi)/2,np.sqrt(pi)/2,r"$2\sqrt{\pi}$",color="black",rotation=0)
# ax.text(np.sqrt(pi)/2,-np.sqrt(pi)/4,r"$2\sqrt{\pi}$",color="black",rotation=0)
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/Wigner_{dim}_0")
# print(fidelity(GKP(2,0,delta,dim).state,GKP(2,1,delta,dim).state))


# Hadamar gate (e^{i pi/4 a.dag a})


a = destroy(dim)
# X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
# Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
# Ds_labels = ["X","Y","Z","Sx","Sp"]
# H = a.dag()*a


# wigner function of d_GKP and some gate on it, with marginals
_,_,d_osc = get_d_gkp(2,0,delta,dim)  # oscillator with d states
# W = WignerDistribution(d_osc, extent=[[-7.5, 7.5], [-7.5, 7.5]])
# Wx = W.marginal(dim=0)
# Wy = W.marginal(dim=1)
# W.visualize()
# Wx.visualize()
# Wy.visualize()
# fig, ax = plot_wigner(d_osc)
# ax.text(-6,6,rf"$\Delta = {delta}$",fontsize='xx-large')
# ax.text(-6,5,rf"$N = {dim}$",fontsize='xx-large')
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/GKP_{dim}")


n = a.dag()*a
H = n**2
tlist = np.linspace(0,pi/16,10)
options = Options(store_states=True)  # get states even if e_ops are calculated
out = mesolve(H, d_osc, tlist, [], [])
fig, ax = plot_wigner(out.states[-1])
ax.text(-6,6.5,rf"$\Delta = {delta}$",fontsize='xx-large')
ax.text(-6,5.6,rf"$N = {dim}$",fontsize='xx-large')
ax.text(-6,4.6,r"$U = e^{i\frac{\pi}{16}\hat{n}^2}$",fontsize='xx-large')
# # mesuring dimension of grid
# ax.plot([0,np.sqrt(pi/2)],[0,np.sqrt(pi/2)],'-',lw=1.5,color="black")
# ax.text(np.sqrt(pi/2)/2,np.sqrt(pi/2)/2-0.5,r"$\sqrt{\pi}$",color="black",rotation=45)
#
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/somegate_GKP_{dim}")


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
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/H_Davg_{dim}")


# H gate


# n = a.dag()*a
# H = n*n
#
# tf = pi/8
# tlist = np.linspace(0,tf,70)  # times for animation
# # tlist = np.linspace(0,tf,1200)  # times for expectation values
# options = Options(store_states=True,nsteps=2500)  # get states even if e_ops are calculated
# osc = GKP(2,0,delta,dim).state
# outs = [mesolve(-H, osc, tlist, [], [D],options=options) for D in Ds]


# sqrtH Wigner function


# fig, ax = plot_wigner(outs[0].states[-1])
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# ax.plot([0,0],[0,np.sqrt(pi)],'-',lw=1.5,color="black")
# ax.text(0.2,np.sqrt(pi)/2,r"$\sqrt{\pi}$",color="black",rotation=0)
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/sqrtHadamar_{dim}")


# sqrtH Wigner function animation


# debut = np.zeros(4,dtype=int).tolist()
# fin = (tf*np.ones(8)).tolist()
# gif_tlist = debut+tlist.tolist()+fin  # times (0 to 3pi/8) with supplementary 0s and 3pi/8s for gif
# frames = np.arange(0,len(tlist),1).tolist()  # index (0,1,2,3,...)
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
#     ax.text(0.02,0.75,rf"$t = {round(gif_tlist[i]/(tf),4)}$"+r"$\frac{\pi}{2}$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.8,r"$H = n^2$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.85,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
#     ax.text(0.02,0.9,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
#

# anim = FuncAnimation(
#     fig, animate, interval=100, frames=frames)
# anim.save("/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/"
#           "Wigner_animation_H_test.gif")


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
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/sqrtH_Davg_{dim}")


# chi function
# j = 0
# delta = 0.25
# dim = 75
# qubit = GKP(2,j,delta,dim)
# rho = qubit.state*qubit.state.dag()
# chi_l = chi_function(rho,7.5)
# plot_chi(chi_l)
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/chi_qubit_j={j}")


# Kraus state
# Kobj = KrausGKP(2,0,0.25,75)
# Kraus_state, delta, dim = Kobj.state, Kobj.delta,Kobj.hilbert_dim
# W = WignerDistribution(Kraus_state, extent=[[-7.5, 7.5], [-7.5, 7.5]])
# Wx = W.marginal(dim=0)
# Wy = W.marginal(dim=1)
# W.visualize()
# Wx.visualize()
# Wy.visualize()
# fig, ax = plot_wigner(Kraus_state)
# ax.text(-6,6,rf"$\Delta = {delta}$")
# ax.text(-6,5,rf"$N = {dim}$")
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/krausGKP_{dim}")
#
# a = destroy(dim)
# X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
# Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
# Ds_labels = ["X","Y","Z","Sx","Sp"]
# H = a.dag()*a
# n = a.dag()*a
# H = n**2
# tlist = np.linspace(0,pi/8,10)
# options = Options(store_states=True)  # get states even if e_ops are calculated
# out = mesolve(H, Kraus_state, tlist, [], [])
# fig, ax = plot_wigner(out.states[-1])
# ax.text(-6,6.5,rf"$\Delta = {delta}$")
# ax.text(-6,5.8,rf"$N = {dim}$")
# ax.text(-6,5.0,r"$U = e^{i\frac{\pi}{16}n^2}$")
# # mesuring dimension of grid
# ax.plot([0,np.sqrt(pi/2)],[0,np.sqrt(pi/2)],'-',lw=1.5,color="black")
# ax.text(np.sqrt(pi/2)/2,np.sqrt(pi/2)/2-0.5,r"$\sqrt{\pi}$",color="black",rotation=45)
#
# plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Basic_GKP_operations/figs/somegate_krausGKP_{dim}")
