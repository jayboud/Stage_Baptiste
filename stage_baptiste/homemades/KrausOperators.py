"""
KrausOperator.py is a code that
was made by Dr. Royer's team.
"""
# -*- coding: utf-8 -*-
from qutip import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import tqdm
import math


pi = np.pi

font = {'family': 'Times New Roman',
        'weight': 'normal',
        # 'fontname':'Times New Roman',
        'size': 16}
# matplotlib.rc('font', **font)

# Base operators
# cavity operators
Ncav = 75
a = destroy(Ncav)
ad = dag(a)
n = ad*a
idCav = qeye(Ncav)

# Function to create Kraus operators
# There are 2 stabilizers (x and p), 
# and for each of them there are 2 measurement results (g and e)
# There are therefore 4 Kraus operators, which are returned as 
# Klist = [[Kxg,Kxe],[Kpg,Kpe]]
# Measuring "g" means no errors, measuring "e" means one error


def opListsBs2(env,latticeGens):
    Klist = []
    for i in range(2):
        latGen = latticeGens[i]
        dpie = displace(Ncav,1j*env*latGen/4/np.sqrt(2))
        dmie = dpie.dag()
        dpl = displace(Ncav,latGen/2/np.sqrt(2))
        dml = dpl.dag()
        deg = dpl*(dpie - 1j*dmie)/2
        dee = dml*(dpie + 1j*dmie)/2
        Klistg = (dpie*(deg + dee) + 1j*dmie*(deg - dee))/2
        Kliste = (dpie*(deg + dee) - 1j*dmie*(deg - dee))/2
        Klist.append([Klistg,Kliste])
        # Klist[2*i,1] = (dmie*(deg + dee) + 1im*dpie*(deg - dee))/2
        # Klist[2*i,2] = (dmie*(deg + dee) - 1im*dpie*(deg - dee))/2
    return Klist


env = 0.15    # Sets the size of the GKP
qubitSquareCodeLatGen = [2*np.sqrt(pi),2j*np.sqrt(pi)]
Klist = opListsBs2(env,qubitSquareCodeLatGen)


# define big spin
Kgg = Klist[0][0]*Klist[1][0]

xvec = np.linspace(-5,5,200)


psi = basis(Ncav)  # Test by starting from vacuum, always measuring g
fig, axes = plt.subplots(3, 3, figsize=(12,12))
for i in range(9):
    psi = (Kgg*psi).unit()  # Important to normalize the state as this is not a unitary evolution
    Wpsi = wigner(psi, xvec, xvec)
    wlim = abs(Wpsi).max()
    axes[i%3,i//3].contourf(xvec, xvec, Wpsi, 100, norm=mpl.colors.Normalize(-wlim, wlim),cmap=mpl.colormaps['RdBu'])
plt.savefig("kraus")
