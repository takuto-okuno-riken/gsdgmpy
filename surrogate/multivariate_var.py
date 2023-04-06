# -*- coding: utf-8 -*-
##
# Surrogate multivariate signal generation by multivariate VAR
# based on autoregressive (AR) surrogates (R. Liegeois et al., 2017)
# returns surrogated signals (Y)
# input:
#  X            multivariate time series matrix (node x time series)
#  net          mVAR network
#  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
#  nodeControl  node control matrix (node x node) (optional)
#  exControl    exogenous input control matrix for each node (node x exogenous input) (optional)
#  dist         distribution of noise to yield surrogates ('gaussian'(default), 'residuals')
#  surrNum      output number of surrogate samples (default:1)
#  yRange       range of Y value (default:[Xmin-Xrange/5, Xmax+Xrange/5])

from __future__ import print_function, division

import numpy as np


def calc(x, net, ex_signal=[], node_control=[], ex_control=[], dist='gaussian', surr_num=1, y_range=np.nan):
    if len(x.shape) == 1:
        x = x.reshape(1, x.shape[0])
    node_num = x.shape[0]
    sig_len = x.shape[1]
    if len(ex_signal):
        ex_num = ex_signal.shape[0]
        x = np.concatenate([x, ex_signal], 0)
    else:
        ex_num = 0
    input_num = node_num + ex_num
    lags = net.lags

    # set control matrix
    control = np.ones((node_num, lags * input_num))
    if len(node_control) == 0:
        node_control = np.ones((node_num, node_num))
    if len(ex_control) == 0:
        ex_control = np.ones((node_num, ex_num))
    for p in range(lags):
        control[:, input_num * p:input_num * (p + 1)] = np.concatenate([node_control, ex_control], 1)

    # find y range
    if type(y_range) == float and np.isnan(y_range):
        t = np.max(x)
        d = np.min(x)
        r = t - d
        y_range = [d-r/5, t+r/5]

    # set residual matrix
    rvlen = len(net.residuals[0])
    for i in range(1, node_num):
        if rvlen > len(net.residuals[i]):
            rvlen = len(net.residuals[i])
    err = np.empty((node_num, rvlen))
    err[:] = np.nan
    for i in range(node_num):
        err[i, :] = net.residuals[i][0:rvlen]
    # set coefficient matrix
    c = np.zeros((node_num, input_num*lags+1))
    for i in range(node_num):
        idx = np.where(control[i, :] == 1)
        c[i, idx[0]] = net.lr_objs[i].coef_
        c[i, c.shape[1]-1] = net.lr_objs[i].intercept_
    # set noise matrix
    if dist == 'gaussian':
        m = np.mean(err, axis=1)
        ec = np.cov(err)
        noise = np.random.multivariate_normal(mean=m, cov=ec, size=rvlen).transpose()
    else:
        noise = err
    s2 = np.ones((input_num*lags+1, 1))
    y = np.zeros((node_num, sig_len, surr_num))
    for k in range(surr_num):
        print('var surrogate sample : '+str(k+1))
        s = x.copy()  # need to care original memory
        st = int(np.random.rand() * (sig_len-lags-1))
        s[0:node_num, 0:lags] = s[0:node_num, st:st+lags]  # initialization of the AR surrogate
        perm2 = np.random.permutation(np.arange(noise.shape[1]))

        for t in range(lags, sig_len):
            a = s[:, t]  # next output
            for p in range(lags):
                s2[input_num * p:input_num * (p + 1), 0] = s[:, t-p-1]
            a[0:node_num] = np.dot(c, s2).reshape((node_num,)) + noise[:, perm2[t-lags]]
            # fixed over shoot values
            if len(y_range) > 0:
                a[a < y_range[0]] = y_range[0]
                a[a > y_range[1]] = y_range[1]
            s[:, t] = a
        y[:, :, k] = s[0:node_num, :]
    return y
