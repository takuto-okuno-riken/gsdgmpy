# -*- coding: utf-8 -*-
##
# Surrogate multivariate signal generation by Fourier Transform (FT)
# returns surrogated signals Y (node x time seriese x surrNum)
# D. Prichard and J. Theiler, Generating surrogate data for time series with several simultaneously measured variables, Phys. Rev. Lett. 73, 951 (1994).
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

    y = np.zeros((node_num, sig_len, surr_num), dtype=x.dtype)
    for k in range(surr_num):
        if (sig_len % 2) == 0:
            r = np.exp(2j * math.pi * np.random.rand(int(sig_len / 2 - 1)))
            rf = np.flip(np.conjugate(r))
            v = np.concatenate([[1], r, [1], rf], 0)
        else:
            r = np.exp(2j * math.pi * np.random.rand(int((sig_len - 1) / 2)))
            rf = np.flip(np.conjugate(r))
            v = np.concatenate([[1], r, rf], 0)

        for i in range(node_num):
            xf = np.fft.fft(x[i, :])
            z = np.fft.ifft(xf * v)
            y[i, :, k] = z.real  # hmm...
    return y
