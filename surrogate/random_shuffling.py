# -*- coding: utf-8 -*-
##
# Surrogate univariate signal generation by Random Shuffling (RS)
# input and output is multivariate, but it is processed by univariate.
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

    y = np.zeros((node_num, sig_len, surr_num), dtype=x.dtype)
    for k in range(surr_num):
        for i in range(node_num):
            xi = x[i, :]
#            si = pd.Series(xi)
            perm = np.random.permutation(range(sig_len))
            y[i, :, k] = xi[perm]
    return y
