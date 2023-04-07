# -*- coding: utf-8 -*-
##
# discriminating statistic function to check IID (independent and identically distributed) for surrogate rank test
# returns discriminating statistic value (t)
# based on J.Theilear and D.Prichard, Physica D: Nonlinear Phenomena (1996) pp.221-235.
# input:
#  X          time series vector
#  params     discriminating statistic function parameters

from __future__ import print_function, division

import numpy as np


def calc(x, params):
    n = len(x[:])
    m = np.mean(x[:])
    x0 = x[0:n-1] - m
    x1 = x[1:n] - m
    x2 = np.dot(x0.reshape(1, n-1), x1.reshape(n-1, 1)) / (n-1)
    m2 = np.mean((x - m) * (x - m))
    t = x2 / m2
    return t
