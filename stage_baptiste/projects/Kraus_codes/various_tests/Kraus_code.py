"""
Author : Jeremie Boudreault
Date: 28/06/2023

A code to analyse errors with Kraus formalism.
"""
import numpy as np
from qutip import *
from scipy.constants import pi
from stage_baptiste.homemades.finite_GKP import GKP
from stage_baptiste.homemades.general_funcs import *
from stage_baptiste.homemades.KrausOperator_JV import *
import matplotlib as mpl
import matplotlib.pyplot as plt

d = 4
j = 0
delta = 0.15
dim = 75
qubit = GKP(2,j,delta,dim)
rho = qubit.state*qubit.state.dag()


# getting useful ops
a = destroy(dim)  # annihilation operator
n_op = a.dag()*a  # number operator
idCav = qeye(dim)  # ideal cavity
Klist = opListsBs2(qubit)  # error operators


# define errors
Kgg = Klist[0][0]*Klist[1][0]
Kge = Klist[0][0]*Klist[1][1]
Keg = Klist[0][1]*Klist[1][0]
Kee = Klist[0][1]*Klist[1][1]
errors_list = [Kgg,Kge,Keg,Kee]

xvec = np.linspace(-5,5,200)


# ---rotating qubit state in d basis so it matches the qubit basis---
# qubit_in_d4_state = (GKP(d,0,delta,dim).state+np.exp(1j*1*pi/4)*GKP(d,1,delta,dim).state-GKP(d,2,delta,dim).state+np.exp(1j*1*pi/4)*GKP(d,3,delta,dim).state).unit()
# rotlist = np.linspace(0,7*pi/4,5)
# rot = mesolve(n_op, qubit_in_d4_state, rotlist, [], [])
# psi = rot.states[-1]
# -------------------------------------------------------------------
psi = qubit.state
p_errors_list = [error_prob(error,psi) for error in errors_list]
# psi = basis(Ncav)  # Test by starting from vacuum, always measuring g
fig, axes = plt.subplots(3, 3, figsize=(12,12))
for i in range(9):
    psi = (Kgg*psi).unit()  # Important to normalize the state as this is not a unitary evolution
    Wpsi = wigner(psi, xvec, xvec)
    wlim = abs(Wpsi).max()
    axes[i%3,i//3].contourf(xvec, xvec, Wpsi, 100, norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
plt.suptitle(r"$|0\rangle_{(2)}$",fontsize='xx-large')
# plt.suptitle(r"$|0\rangle_{(2)}$ in basis $d=4,m=n=1$",fontsize='xx-large')
plt.savefig(f"/Users/jeremie/Desktop/Stage_Baptiste/stage_baptiste/projects/Kraus_codes/various_tests/figs/Kraus_d=2,j=0")
