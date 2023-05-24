"""
Author : Jeremie Boudreault
Date: 11/05/2022

Code that produces a more detailed animation to inspect intermediate
states during the application of the sqrtH gate.
"""

import numpy as np
from qutip import *
from stage_baptiste.homemades.finite_GKP import get_gkp, GKP
from scipy.constants import pi
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


delta = 0.25
dim = 100
osc = GKP(delta,dim).state

a = destroy(dim)
n = a.dag()*a
H = n*n
tf = pi/8
t_mid = pi/16
final = t_mid
iter = 501
tlist = np.linspace(0,final+final/iter,iter)  # for precised animation

# e_ops of needed
X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
Ds_labels = ["X","Y","Z","Sx","Sp"]


out = mesolve(H, osc, tlist, [], [])


# sqrtH Wigner function animation

frames = np.arange(0,len(tlist),1).tolist()+(range(len(tlist))[-1]*np.ones(4,dtype=int)).tolist()  # index (0,1,2,3,...)
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
ax.set_title("Wigner function", fontsize=12)
xvec = np.linspace(-7.5, 7.5, 200)
W0 = wigner(out.states[0], xvec, xvec)  # initial plot for the animation
W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
wlim = abs(W).max()
cax = ax.contourf(xvec, yvec, W, 100,norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])


def animate(i):
    ax.clear()
    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
    ax.set_title("Wigner function", fontsize=12)
    W0 = wigner(out.states[i], xvec, xvec)
    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
    ax.contourf(xvec, yvec, W, 100, norm=mpl.colors.Normalize(-wlim, wlim), cmap=mpl.colormaps['RdBu'])
    ax.text(0.02,0.75,rf"$t = {round(tlist[i],4)}$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.8,r"$H = n^2$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.85,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.9,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)


anim = FuncAnimation(
    fig, animate, interval=100, frames=frames)
anim.save("/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Pieceable_sqrtH/figs/"
          f"Detailed_animation_mid_sqrtH_it={iter}.gif")
