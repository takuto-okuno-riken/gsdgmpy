# -*- coding: utf-8 -*-
##
# Calculate partial auto-correlation (simple version)
# returns matrix (node x lags)
# input:
#  X                multivariate time series matrix (node x time series)
#  max_lag          time lags for Partial Auto-Correlation function (default: 15)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def calc(x, max_lag=15):
    node_num = x.shape[0]
    xr = np.zeros((node_num, max_lag+1), dtype=x.dtype)
    for i in range(node_num):
        xr[i, :] = sm.tsa.stattools.pacf(x[i, :], nlags=max_lag, method="ols")
    return xr


def plot(x, max_lag=15):
    xr = calc(x=x, max_lag=max_lag)
    plt.matshow(xr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Source Nodes')
    plt.ylabel('Target Nodes')
    plt.show(block=False)
    plt.pause(1)
    return xr
