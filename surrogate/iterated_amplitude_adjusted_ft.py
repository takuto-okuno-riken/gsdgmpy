# -*- coding: utf-8 -*-
##
# Surrogate univariate signal generation by Iterated Amplitude adjusted Fourier Transform (IAAFT)
# input and output is multivariate, but it is processed by univariate.
# returns surrogated signals (Y)
# input:
#  X          multivariate time series matrix (node x time series)
#  maxIter    maximum iteration number (default:100)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import numpy as np
from surrogate.amplitude_adjusted_ft import calc_uni as amplitude_adjusted_ft_uni


def calc(x, max_iter=100, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    y = np.zeros((node_num, sig_len, surr_num))
    for k in range(surr_num):
        for i in range(node_num):
            xi = x[i, :]
            y[i, :, k] = calcUni(x=xi, max_iter=max_iter)
    return y


def calcUni(x, max_iter):
    x2 = np.sort(x)
    s = np.abs(np.fft.fft(x))
    y = amplitude_adjusted_ft_uni(x=x)
    for i in range(max_iter):
        r = np.fft.fft(y)
        sm = (r / np.abs(r)) * s
        h = np.abs(np.fft.ifft(sm))
        kk = np.argsort(h)
        ll = np.argsort(kk)
        y2 = x2[ll]
        if np.all(y == y2):
            break
        y = y2
    return y
