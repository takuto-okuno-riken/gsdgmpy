# -*- coding: utf-8 -*-
##
# Calculate cross correlation matrix (simple version)
# returns 3D matrix (node x node x max_lag*2+1)
# input:
#  X         multivariate time series matrix (node x time series)
#  max_lag   time lag number (default: 5)

from __future__ import print_function, division

import timeit
import numpy as np
import scipy
import matplotlib.pyplot as plt


def calc(x, max_lag=5):
    node_num = x.shape[0]
    sig_len = x.shape[1]
    xm = np.mean(x, axis=1).reshape((node_num, 1))
    x2 = x - np.repeat(xm, sig_len, axis=1)
    xn = np.linalg.norm(x2, ord=2, axis=1)
    ccm = np.zeros((node_num, node_num, max_lag*2+1), dtype=x.dtype)
    tic = timeit.default_timer()
    print('start to calc ccm')

    # check all same value or not
    ulen = np.zeros(node_num)
    for i in range(node_num):
        ulen[i] = np.unique(x[i, :]).size

    for i in range(node_num):
        xi = x2[i, :]
        for j in range(i, node_num):
            if ulen[i] == 1 and ulen[j] == 1:
                ccm[i, j, :] = 0  # for flat line bug (to compatible with matlab ver)
                continue

            xj = x2[j, :]
            cc = scipy.signal.correlate(xi, xj, mode='full')
            cc /= xn[i] * xn[j]
            ccm[i, j, :] = cc[(sig_len-1-max_lag):(sig_len+max_lag)]

    e = np.ones((node_num, node_num), dtype=x.dtype)
    mask = np.tril(e, k=-1)
    for p in range(-max_lag, max_lag):
        b = ccm[:, :, max_lag - p]
        b = b.transpose() * mask
        ccm[:, :, max_lag + p] += b
    toc = timeit.default_timer()
    print('done t='+str(toc-tic))
    return ccm


def plot(x, max_lag=5):
    ccm = calc(x=x, max_lag=max_lag)
    plt.matshow(ccm[:, :, max_lag], vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Source Nodes')
    plt.ylabel('Target Nodes')
    plt.show(block=False)
    plt.pause(1)
    return ccm
