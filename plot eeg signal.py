# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:52:19 2020

@author: antho
"""
X_first = data_filter
n_samples, n_rows = 1750, 22
t = 10 * np.arange(n_samples) / n_samples

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Plot the EEG
fig = plt.figure("EEG of Feet MI")

ticklocs = []
ax2 = fig.add_subplot(1,1,1)
ax2.set_xlim(0, 7)
ax2.set_xticks(np.arange(7))

dmin = X_first.min()
dmax = X_first.max()

dr = (dmax - dmin) * 0.7  # Crowd them a bit.
y0 = dmin
y1 = (n_rows - 1) * dr + dmax
ax2.set_ylim(y0, y1)

segs = []
for i in range(n_rows):
    segs.append(np.column_stack((t, X_first[i, :])))
    ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None)
ax2.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax2.set_yticks(ticklocs)
ax2.set_yticklabels(['Fz', '0', '1', '2', '3', '4', '5', 'C3', '6', 'Cz', 
                     '7', 'C4', '8', '9', '10', '11', '12', '13', '14', 'Pz',
                     '21', '22'])

ax2.set_xlabel('Time (s)')
ax2.set_title('EEG signal of Feet MI (after bandpassed filtering)')
