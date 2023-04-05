# -*- coding: utf-8 -*-
##
# Calculate Cosign similarity
# returns one value in range [-1, 1]
# input:
#  x         any matrix
#  y         any matrix

from __future__ import print_function, division

import numpy as np


def calc(x, y):
    x2 = x[~np.isnan(x) & ~np.isnan(y)]
    y2 = y[~np.isnan(x) & ~np.isnan(y)]
    c = np.dot(x2, y2)
    if np.any(x2 != 0) and np.any(y2 != 0):
        s = c / np.sqrt(np.dot(x2, x2) * np.dot(y2, y2))
    else:
        s = 0
    return s
