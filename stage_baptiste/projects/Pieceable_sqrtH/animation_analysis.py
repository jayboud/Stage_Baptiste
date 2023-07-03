"""
Author : Jeremie Boudreault
Date: 11/05/2022

Code that analyses in more details animations to inspect
intermediate states during the application of the sqrtH gate.
"""

import numpy as np
from qutip import *
from qutip.wigner import _wigner_clenshaw
from stage_baptiste.homemades.finite_GKP import get_d_gkp, GKP
from stage_baptiste.homemades.general_funcs import rot_wigner_clenshaw
from scipy.constants import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

delta = 0.25
dim = 100
osc = GKP(2,0,delta,dim).state

a = destroy(dim)

# e_ops of needed
X,Y,Z,Sx,Sp = np.sqrt(pi/2),np.sqrt(pi/2)*(1+1j),np.sqrt(pi/2)*1j,np.sqrt(2*pi),np.sqrt(2*pi)*1j
Ds = [displace(dim,gamma) for gamma in [X,Y,Z,Sx,Sp]]  # X,Z,Y,Sx,Sp
Ds_labels = ["X","Y","Z","Sx","Sp"]


# exp(i*n*pi/16) with mod2/mod4 states

_,_,d_osc = get_d_gkp(2,0,delta,dim)  # oscillator with d states
n = a.dag()*a
H = n**2
tlist = np.linspace(0,pi/16,10)
options = Options(store_states=True)  # get states even if e_ops are calculated
out = mesolve(H, d_osc, tlist, [], [])
psi = out.states[-1]
if psi.type == 'ket' or psi.type == 'bra':
    rho = ket2dm(psi)
else:
    rho = psi
# ----- Wigner function  ---------
# angle = pi/4  # anticlockwise rotation
angle = 0  # anticlockwise rotation
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


W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)
wlim = abs(W).max()
fig = plt.figure(figsize=(8,8))
fig.suptitle(r"Wigner Function")
# fig.suptitle(r"Wigner Function and its marginals")
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
ax1 = fig.add_subplot(gs[0])
ax1.set_box_aspect(1)
ax1.contourf(xvec, yvec, rotW0, 100,norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
ax2 = fig.add_subplot(gs[1],sharey=ax1)
# ax2.plot(Wy.data,np.linspace(-7.5,7.5,250))
ax2.plot(rotWy,np.linspace(-7.5,7.5,200)[:,None])
ax3 = fig.add_subplot(gs[2],sharex=ax1)
# ax3.plot(np.linspace(-7.5,7.5,250),Wx.data)
ax3.plot(np.linspace(-7.5,7.5,200)[:,None],rotWx)
ax1.text(-6,6.5,rf"$\Delta = {delta}$")
ax1.text(-6,5.8,rf"$N = {dim}$")
ax1.text(-6,5.0,r"$U = e^{i\frac{\pi}{16}n^2}$")
# mesuring dimension of grid
ax1.plot([0,0],[0,np.sqrt(pi)],'-',lw=1.5,color="black")
ax1.text(np.sqrt(pi/2)/4,np.sqrt(pi)/2,r"$\sqrt{\pi}$",color="black",rotation=0)

plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Pieceable_sqrtH/figs/margin_GKP_{dim},t=pi16")


# diagonal plot (to join eventually)


# scales = (-7.5, 7.5,0, 0.04)
# scale_factor = 80
# deg_angle = 0
# t = Affine2D().scale(1,scale_factor).rotate_deg(deg_angle).translate(0,0)
# h = floating_axes.GridHelperCurveLinear(t,scales)
# ax = floating_axes.FloatingSubplot(fig, gs[0], grid_helper=h)
# diag_marg = np.concatenate((np.linspace(-7.5,7.5,200)[:,None],rotW0.mean(axis=1)[:,None]),axis=1)
# rot_diag_marg = Affine2D().rotate_deg(deg_angle).transform(diag_marg)
# ax.plot(rot_diag_marg[:,0],scale_factor*rot_diag_marg[:,1]/4)
# diag_ax = fig.add_axes(ax)


# middle state

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

plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Pieceable_sqrtH/figs/middle_GKP")


