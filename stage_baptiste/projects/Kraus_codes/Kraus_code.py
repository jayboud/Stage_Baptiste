"""
Author : Jeremie Boudreault
Date: 28/06/2023

A code to analyse errors with Kraus formalism.
"""
import numpy as np
from qutip import *
from scipy.constants import pi
from stage_baptiste.homemades.finite_GKP import GKP
from stage_baptiste.homemades.KrausOperator_JV import opListsBs2
import matplotlib as mpl
import matplotlib.pyplot as plt


delta = 0.15
dim = 75
qubit = GKP(2,0,delta,dim)

a = destroy(dim)
ad = dag(a)
n = ad*a
idCav = qeye(dim)
Klist = opListsBs2(qubit)  # getting the list of operators


# define big spin
Kgg = Klist[0][0]*Klist[1][0]
Kge = Klist[0][0]*Klist[1][1]
Keg = Klist[0][1]*Klist[1][0]
Kee = Klist[0][1]*Klist[1][1]

xvec = np.linspace(-5,5,200)


# psi = basis(Ncav)  # Test by starting from vacuum, always measuring g
psi = qubit.state  # Test by starting from vacuum, always measuring g
fig, axes = plt.subplots(3, 3, figsize=(12,12))
for i in range(9):
    psi = (Kgg*psi).unit()  # Important to normalize the state as this is not a unitary evolution
    Wpsi = wigner(psi, xvec, xvec)
    wlim = abs(Wpsi).max()
    axes[i%3,i//3].contourf(xvec, xvec, Wpsi, 100, norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
plt.savefig("kraus_JV")