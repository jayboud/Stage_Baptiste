import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes


fig = plt.figure(figsize=(8,8))
fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

gs = GridSpec(3, 3, width_ratios=[4, 4, 1], height_ratios=[4, 4, 1])
ax1 = fig.add_subplot(gs[4])
ax1.set_box_aspect(1)
ax2 = fig.add_subplot(gs[5])
ax3 = fig.add_subplot(gs[7])

scales = (0, 5, 0, 5)
t = Affine2D().scale(4,1).rotate_deg(45)
h = floating_axes.GridHelperCurveLinear(t, scales)
ax = floating_axes.FloatingSubplot(fig, gs[0], grid_helper=h)
diag_ax = fig.add_axes(ax)

plt.savefig("graphtest")

