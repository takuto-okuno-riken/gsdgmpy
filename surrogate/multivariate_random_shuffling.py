# -*- coding: utf-8 -*-
##
# Surrogate multivariate signal generation by Random Shuffling (RS)
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

    y = np.zeros((node_num, sig_len, surr_num))
    for k in range(surr_num):
        perm = np.random.permutation(range(sig_len))
        for i in range(node_num):
            xi = x[i, :]
            y[i, :, k] = xi[perm]
    return y

