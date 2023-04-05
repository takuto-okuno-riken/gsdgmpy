# -*- coding: utf-8 -*-
##
# Surrogate multivariate signal generation by Iterated Amplitude adjusted Fourier Transform (IAAFT)
# returns surrogated signals Y (node x time seriese x surrNum)
# input:
#  X          multivariate time series matrix (node x time series)
#  maxIter    maximum iteration number (default:100)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import numpy as np
import surrogate


def calc(x, max_iter=100, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    x2 = np.sort(x, axis=1)
    s = np.abs(np.fft.fft(x, axis=1))
    y = surrogate.multivariate_amplitude_adjusted_ft(x=x, surr_num=surr_num)
    for k in range(surr_num):
        yk = y[:, :, k]
        for i in range(max_iter):
            r = np.fft.fft(yk, axis=1)
            sm = (r / np.abs(r)) * s
            h = np.abs(np.fft.ifft(sm, axis=1))
            kk = np.argsort(h, axis=1)
            ll = np.argsort(kk, axis=1)
            yk2 = np.zeros((node_num, sig_len))
            for j in range(node_num):
                t = x2[j, :]
                yk2[j, :] = t[ll[j, :]]
            if np.all(yk == yk2):
                break
            yk = yk2
    return y

