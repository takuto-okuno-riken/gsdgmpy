# -*- coding: utf-8 -*-
##
# Caluclate surrogate rank test
# returns significance vector (H=1 or 0), p-value vector (P), discriminating statistic matrix T (node x surrNum) and original rank vector(Rank).
# input:
#  X                original multivariate time series matrix (node x time series)
#  Y                surrogate multivariate time series matrix (node x time series x surrNum)
#  statisticFunc    discriminating statistic function
#  statisticParams  discriminating statistic function parameters
#  side             bottm-side(1), both-side(2), top-side(3) (default:2)
#  alpha            the significance level of statistic (default:0.05)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

def calc(x, cy, func, params=[], side=2, alpha=0.05):
    node_num = x.shape[0]
    surr_num = len(cy)
    t = np.empty((node_num, surr_num+1), dtype=x.dtype)

    # calculate discriminating statistic
    for i in range(node_num):
        t[i, 0] = func.calc(x[i, :], params)
        for k in range(surr_num):
            t[i, k+1] = func.calc(cy[k][i, :], params)

    # rank test
    I = np.argsort(t, axis=1) + 1
    r = np.repeat(np.arange(1, surr_num+2).reshape(1, surr_num+1), node_num, axis=0)
    I[I != 1] = 0
    r2 = r * I
    rank = np.sum(r2, axis=1)
    if side == 1:
        p = rank / (surr_num+1)
    elif side == 3:
        p = 1 - ((rank - 1) / (surr_num + 1))
    else:
        r3 = rank
        n = (surr_num + 1) / 2
        idx = np.where(r3 > n)
        r3[idx[0]] = (surr_num + 2) - r3[idx[0]]
        p = r3 / n
    h = p < alpha
    return h, p, t, rank


def plot(p, t, rank, name):
    node_num = t.shape[0]
    surr_num = t.shape[1]
    for i in range(node_num):
        plt.figure()
        for k in range(1, surr_num):
            plt.plot([t[i, k], t[i, k]], [0, 0.5], 'b', linewidth=0.6)
        plt.plot([t[i, 0], t[i, 0]], [0, 1], 'r', linewidth=0.6)
        plt.title(name+' surrogate test, Node '+str(i+1)+' : rank='+str(rank[i])+', p-value='+str(p[i]))
        plt.show(block=False)
        plt.pause(1)
