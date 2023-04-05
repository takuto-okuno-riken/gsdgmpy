# -*- coding: utf-8 -*-
##
# Calculate partial correlation matrix (simple version)
# returns matrix (node x node)
# input:
#  X         multivariate time series matrix (node x time series)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def calc(x):
    node_num = x.shape[0]
    pcm = np.zeros((node_num, node_num))

    lr = LinearRegression(fit_intercept=True)
    for i in range(node_num):
        xi = x[i, :].transpose()
        for j in range(i, node_num):
            idx = np.arange(node_num)
            if j > i:
                idx = np.delete(idx, j)
            idx = np.delete(idx, i)
            xtj = x[idx, :].transpose()
            xj = x[j, :].transpose()

            lr.fit(xtj, xi)
            pred = lr.predict(xtj)
            r1 = (xi - pred)
            lr.fit(xtj, xj)
            pred = lr.predict(xtj)
            r2 = (xj - pred)

            cr = np.corrcoef(x=r1, y=r2)
            pcm[i, j] = cr[0, 1]
            pcm[j, i] = cr[0, 1]

    return pcm


def plot(x):
    xr = calc(x=x)
    plt.matshow(xr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Source Nodes')
    plt.ylabel('Target Nodes')
    plt.show(block=False)
    plt.pause(1)
    return xr
