# -*- coding: utf-8 -*-
##
# Surrogate univariate signal generation by Fourier Transform (FT)
# input and output is multivariate, but it is processed by univariate.
# returns surrogated signals Y (node x time seriese x surrNum)
# input:
#  X          multivariate time series matrix (node x time series)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import math
import numpy as np


def calc(x, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    y = np.zeros((node_num, sig_len, surr_num))
    for k in range(surr_num):
        for i in range(node_num):
            xi = x[i, :]
            y[i, :, k] = calc_uni(xi)
    return y


def calc_uni(x):
    n = len(x)
    if (n % 2) == 0:
        r = np.exp(2j * math.pi * np.random.rand(int(n/2 - 1)))
        rf = np.flip(np.conjugate(r))
        v = np.concatenate([[1], r, [1], rf], 0)
    else:
        r = np.exp(2j * math.pi * np.random.rand(int((n-1)/2)))
        rf = np.flip(np.conjugate(r))
        v = np.concatenate([[1], r, rf], 0)

    xf = np.fft.fft(x)
    y = np.fft.ifft(xf * v)
    y = y.real  # hmm...
    return y
