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
    m = np.mean(x[:])
    x2 = x - m
    m2 = np.mean(x2 * x2 * x2 * x2)
    m3 = np.mean(x2 * x2)
    t = m2 / (m3 * m3)
    return t
