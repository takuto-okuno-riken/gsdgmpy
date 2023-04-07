# -*- coding: utf-8 -*-
##
# Surrogate multivariate signal generation by Random Gaussian (RG)
# returns surrogated signals Y (node x time seriese x surrNum)
# input:
#  X          multivariate time series matrix (node x time series)
#  surr_num   output number of surrogate samples (default:1)

from __future__ import print_function, division

import numpy as np


def calc(x, surr_num=1):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]

    m = np.mean(x, axis=1)
    ec = np.cov(x)
    y = np.zeros((node_num, sig_len, surr_num), dtype=x.dtype)
    for k in range(surr_num):
        si = np.random.multivariate_normal(mean=m, cov=ec, size=sig_len)
        y[:, :, k] = np.transpose(si)
    return y

