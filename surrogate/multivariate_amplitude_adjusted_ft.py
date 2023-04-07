# -*- coding: utf-8 -*-
##
#% Surrogate multivariate signal generation by Amplitude adjusted Fourier Transform (AAFT)
#% returns surrogated signals Y (node x time seriese x surrNum)
# input:
#  X          multivariate time series matrix (node x time series)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import numpy as np
import surrogate

def calc(x, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    x2 = np.sort(x, axis=1)
    ii = np.argsort(x, axis=1)
    jj = np.argsort(ii, axis=1)
    r = np.sort(np.random.randn(node_num, sig_len), axis=1)
    g = np.zeros((node_num, sig_len), dtype=x.dtype)
    for i in range(node_num):
        ri = r[i, :]
        g[i, :] = ri[jj[i, :]]

    h = surrogate.multivariate_phase_randomized_ft(g, surr_num)
    y = np.zeros((node_num, sig_len, surr_num), dtype=x.dtype)
    for k in range(surr_num):
        kk = np.argsort(h[:, :, k], axis=1)
        ll = np.argsort(kk, axis=1)
        for i in range(node_num):
            xi = x2[i, :]
            y[i, :, k] = xi[ll[i, :]]
    return y
