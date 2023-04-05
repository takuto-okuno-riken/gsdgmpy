# -*- coding: utf-8 -*-
##
# Calculate Discrete Fourier transform
# returns One-sided amplitude spectrum D (node x sampling spectrum)
# input:
#  X         multivariate time series matrix (node x time series)
#  n         DFT sampling number (even number) (default: 100)
#  Fs        sampling frequency of time seriese (default: 0.5)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


def calc(x, n_dft=100):
    y = np.fft.fft(x, axis=1, n=n_dft)
    p2 = np.abs(y / n_dft)
    if len(x.shape) > 2:
        p1 = p2[:, 0:int(n_dft/2+1), :]
        p1[:, 1:p1.shape[1]-1, :] = 2 * p1[:, 1:p1.shape[1]-1, :]
        d = p1[:, 1:p1.shape[1]-1, :]
    else:
        p1 = p2[:, 0:int(n_dft/2+1)]
        p1[:, 1:p1.shape[1]-1] = 2 * p1[:, 1:p1.shape[1]-1]
        d = p1[:, 1:p1.shape[1]-1]
    return d, p1


def plot(x, n_dft=100, fs=0.5):
    d, p1 = calc(x=x, n_dft=n_dft)
    f = fs * np.arange(0, int(n_dft/2+1)) / n_dft
    for i in range(len(x.shape) - 1):
        if len(x.shape) > 2:
            plt.plot(f, np.transpose(p1[:, :, i]))
        else:
            plt.plot(f, np.transpose(p1))
        plt.title('Single-Sided Amplitude Spectrum of S(t)')
        plt.xlabel('f (Hz)')
        plt.ylabel('|P1(f)|')
        plt.show(block=False)
        plt.pause(1)
    return d, p1
