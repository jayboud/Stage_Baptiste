"""
Author : Jeremie Boudreault
Date: 11/05/2022

Code that compares qubit and qudit d=4 GKPs.
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

wigner = False
displacement = True
d = 4
m,n = 1,1
j = 0
delta = 0.3
dim = 100
osc = 1/2*(GKP(d,0,delta,dim).state+np.exp(1j*1*pi/4)*GKP(d,1,delta,dim).state-GKP(d,2,delta,dim).state+np.exp(1j*1*pi/4)*GKP(d,3,delta,dim).state)
orig_osc = GKP(2,0,delta,dim).state
print(orig_osc.dag()*osc)
# COMMENT FAIRE LA BONNE PROJECTION (ÉTAT EST ROTATÉ)

a = destroy(dim)

# e_ops if needed
X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
Ds_labels = ["X","Y","Z","Sx","Sp"]


n_op = a.dag()*a
H = n_op**2
tlist = np.linspace(0,pi/16,200)
options = Options(store_states=True)  # get states even if e_ops are calculated
outs = mesolve(H, osc, tlist, [], Ds,options=options)
orig_outs = mesolve(H, orig_osc, tlist, [], Ds,options=options)
psi = osc
if psi.type == 'ket' or psi.type == 'bra':
    rho = ket2dm(psi)
else:
    rho = psi



# ----- Wigner function  ---------
angle = pi/4  # anticlockwise rotation
xvec = np.linspace(-7.5, 7.5, 200)
W0 = _wigner_clenshaw(rho, xvec, xvec)  # no rotation
rotW0 = rot_wigner_clenshaw(rho, xvec, xvec,rot=angle)  # with pi/4 rotation
# ------------------------
# ----- marginals ---------
W = WignerDistribution(rho, extent=[[-7.5, 7.5], [-7.5, 7.5]])
Wx, rotWx = W.marginal(dim=0), rotW0.mean(axis=0)[:,None]
Wy, rotWy = W.marginal(dim=1), rotW0.mean(axis=1)[:,None]
# ------------------------


# mise en graphique
if wigner:
    domain = np.arange(-4,4)
    ticks = [i*np.sqrt(pi) for i in domain]
    ticks_name = [rf"{i}$\sqrt\pi$" for i in domain]

    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
    wlim = abs(W).max()
    fig = plt.figure(figsize=(8,8))
    # fig.suptitle(r"$(\pi/4$ rotated) Wigner Function and its marginals")
    fig.suptitle(r"Wigner Function and its marginals")
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(1)
    ax1.set_xticks(ticks,ticks_name)
    ax1.set_yticks(ticks,ticks_name)
    ax1.contourf(xvec, yvec, rotW0, 100,norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
    ax2 = fig.add_subplot(gs[1],sharey=ax1)
    # ax2.plot(Wy.data,np.linspace(-7.5,7.5,250))
    ax2.plot(rotWy,np.linspace(-7.5,7.5,200)[:,None])
    ax3 = fig.add_subplot(gs[2],sharex=ax1)
    # ax3.plot(np.linspace(-7.5,7.5,250),Wx.data)
    ax3.plot(np.linspace(-7.5,7.5,200)[:,None],rotWx)
    ax1.text(-6,6.5,rf"$\Delta = {delta}$")
    ax1.text(-6,5.8,rf"$N = {dim}$")
    # ax1.text(-6,5.0,r"$U = e^{i\frac{\pi}{16}n^2}$")
    # ax1.text(-6,5,r"$|\psi\rangle = |\bar{0}\rangle_{(2)}$")
    ax1.text(-6,5,r"$|\psi\rangle = \frac{1}{2}(|\bar{0}\rangle_{(4)} + e^{i\pi/4}|\bar{1}\rangle_{(4)} -"
                                                r"|\bar{2}\rangle_{(4)} + e^{i\pi/4}|\bar{3}\rangle_{(4)})$")
    # mesuring dimension of grid
    ax1.plot([0,0],[0,np.sqrt(pi)],'-',lw=1.5,color="black")
    ax1.text(np.sqrt(pi/2)/4,np.sqrt(pi)/2,r"$\sqrt{\pi}$",color="black",rotation=0)

    plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/figs/qubit_equiv,j={j},d={d}")


# average of displacements for H
if displacement:
    fig, ax = plt.subplots()
    for result,label in zip(outs.expect,Ds_labels):
        line_re = ax.plot(tlist,np.real(result),label=rf"${label}$")  # real part
        line_im = ax.plot(tlist,np.imag(result),ls="--",color=line_re[0].get_color())  # imaginary part
    ax.text(0.02,0.95,rf"$\Delta = {delta}$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.87,rf"$N = {dim}$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.79,r"$U = e^{\frac{\pi}{16}n^2}$",ha='left', va='top', transform=ax.transAxes)
    ax.text(0.02,0.71,r"$|\psi\rangle = \frac{1}{2}(|\bar{0}\rangle_{(4)} + e^{i\pi/4}|\bar{1}\rangle_{(4)} -"
                                                r"|\bar{2}\rangle_{(4)} + e^{i\pi/4}|\bar{3}\rangle_{(4)})$",ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle D \rangle$",rotation=0)
    plt.title("Valeur moyenne du déplacement en fonction du temps")
    plt.legend(loc="lower right")
    plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Other_GKPs/GKP_4/figs/pion16_Davg_d={d},m={m},n={n}")
