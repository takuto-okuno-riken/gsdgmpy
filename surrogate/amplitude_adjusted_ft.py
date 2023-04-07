# -*- coding: utf-8 -*-
##
# Surrogate univariate signal generation by Amplitude adjusted Fourier Transform (AAFT)
# input and output is multivariate, but it is processed by univariate.
# returns surrogated signals (Y)
# input:
#  X          multivariate time series matrix (node x time series)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import numpy as np
from surrogate.phase_randomized_ft import calc_uni as phase_randomized_ft_uni


def calc(x, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    y = np.zeros((node_num, sig_len, surr_num), dtype=x.dtype)
    for k in range(surr_num):
        for i in range(node_num):
            xi = x[i, :]
            y[i, :, k] = calc_uni(x=xi)
    return y


def calc_uni(x):
    n = len(x)
    x2 = np.sort(x)
    i = np.argsort(x)
    j = np.argsort(i)
    r = np.sort(np.random.randn(n))
    g = r[j]
    h = phase_randomized_ft_uni(x=g)
    k = np.argsort(h)
    ll = np.argsort(k)
    return x2[ll]
