# -*- coding: utf-8 -*-
##
# discriminating statistic function to check gaussian for surrogate rank test
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
    m2 = np.mean((x - m) * (x - m))
    t1 = np.empty((n-2), dtype=x.dtype)
    for i in range(2, n):
        t1[i-2] = x[i] - f(x[i-1], x[i-2], params)
    t2 = np.mean(t1 * t1)
    t = t2 / m2
    return t


def f(u, v, c):
    return c[0] + c[1]*u + c[2]*v + c[3]*u*u + c[4]*u*v + c[5]*v*v
