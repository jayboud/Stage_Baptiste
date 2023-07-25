

import numpy as np
from scipy.constants import pi
from qutip import *
from matplotlib import pyplot as plt

dim = 100
a = destroy(dim)
x_op = (a + a.dag())/np.sqrt(2)
p_op = (a - a.dag())/(1j*np.sqrt(2))
eigs,bins = [],[]
for op in [x_op,p_op]:
    op_eig = np.array(op.eigenenergies())
    eigs.append(op_eig)
    op_bins = np.copy(op_eig)
    condition = abs(abs(op_bins)/np.sqrt(pi) - abs(op_bins)/np.sqrt(pi)//1)
    up_bins, down_bins = condition < 1/2, condition > 1/2
    op_bins[up_bins],op_bins[down_bins] = 1, -1
    bins.append(op_bins)


plt.plot(eigs[0],bins[0])
plt.plot(bins[1],eigs[1])
plt.savefig('bins')
plt.show()
