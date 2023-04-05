# -*- coding: utf-8 -*-
##
# Calculate correlation matrix (simple version)
# returns matrix (node x node)
# input:
#  X         multivariate time series matrix (node x time series)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


def calc(x):
#    xs = np.std(x, axis=1)
#    xcov = np.cov(x)
#    xcs = np.dot(np.reshape(xs, (xs.shape[0], 1)), np.reshape(xs, (1, xs.shape[0])))
#    xr = xcov / xcs
    xr = np.corrcoef(x=x)
    return xr


def plot(x):
    xr = calc(x=x)
    plt.matshow(xr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Source Nodes')
    plt.ylabel('Target Nodes')
    plt.show(block=False)
    plt.pause(1)
    return xr
