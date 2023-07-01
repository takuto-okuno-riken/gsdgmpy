# -*- coding: utf-8 -*-
##
# Calculate multivariate skewness and kurtosis
# returns multivariate skewness (mskew) and multivariate kurtosis (mkurt)
# input:
#  X                multivariate time series matrix (node x time series)

from __future__ import print_function, division

import numpy as np


def calc(x):
    node_num = x.shape[0]
    n = x.shape[1]
    m = np.mean(x, axis=1)
    x2 = np.copy(x)
    for i in range(node_num):
        x2[i, :] = x[i, :] - m[i]
    c = np.cov(x2)
    if np.linalg.det(c) == 0:
        ci = np.linalg.pinv(c)
    else:
        ci = np.linalg.inv(c)
    d = np.transpose(x2) @ ci @ x2

    mskew = np.sum(d * d * d) / (n*n)
    dd = np.diag(d)
    mkurt = dd @ np.transpose(dd) / n
    return mskew, mkurt
