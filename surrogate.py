# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.parse_surrogate_options import ParseOptions
from utils.convert_sigmd import SigmoidConverter
import models
import surrogate


# -------------------------------------------------------------------------
# matrix calculation
def save_result_files(opt, y, outname):
    out_path_f = opt.outpath + os.sep + outname
    if opt.format == 0:  # csv each
        for i in range(y.shape[2]):
            f_name = out_path_f + '_' + str(i+1) + '.csv'
            print('output csv file : ' + f_name)
            np.savetxt(f_name, y[:, :, i], delimiter=',')

    elif opt.format == 1:  # mat each
        for i in range(y.shape[2]):
            f_name = out_path_f + '_' + str(i+1) + '.mat'
            print('output mat file : ' + f_name)
            sio.savemat(f_name, {'X': y[:, :, i]})

    elif opt.format == 2:  # mat all
        f_name = out_path_f + '_all.mat'
        print('output mat file : ' + f_name)
        names = np.empty((y.shape[2]), dtype=object)
        cx = np.empty((y.shape[2]), dtype=object)
        for i in range(y.shape[2]):
            cx[i] = y[:, :, i]
            names[i] = outname+'_'+str(i+1)
        sio.savemat(f_name, {'CX': cx, 'names': names})


# -------------------------------------------------------------------------
# main
if __name__ == '__main__':
    options = ParseOptions()
    opt = options.parse()

    if not opt.multi and not opt.uni:
        opt.multi = True

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string
    if type(opt.noise) is list:
        opt.noise = opt.noise[0]  # replaced by string

    # read time-series
    for i in range(len(opt.in_files)):
        if not os.path.isfile(opt.in_files[i]):
            print('bad file name. ignore : ' + opt.in_files[i])
            continue

        x = []
        print('loading signals : ' + opt.in_files[i])
        name = os.path.splitext(os.path.basename(opt.in_files[i]))[0]
        if '.csv' in opt.in_files[i]:
            csv_input = pd.read_csv(opt.in_files[i], header=None)
            x = csv_input.values
        elif '.mat' in opt.in_files[i]:
            dic = sio.loadmat(opt.in_files[i])
            x = dic.get('X')

        if len(x) == 0 or x is None:
            print('file does not contain data. ignore : ' + opt.in_files[i])
            continue

        if opt.format == 2:
            if i == 0:
                savename = name
        else:
            savename = name

        # convert input & exogenous signals
        if opt.transform == 1:
            conv = SigmoidConverter()
            x, sig, c, max_si, min_si = conv.to_sigmoid_signal(x=x, centroid=opt.transopt)

        # show input signals
        if opt.showsig:
            plt.figure()
            plt.plot(x.transpose(), linewidth=0.3)
            plt.title('Input time-series : ' + name)
            plt.xlabel('Time frames')
            plt.ylabel('Signal value')
            plt.show(block=False)

        plt.pause(1)

        # -------------------------------------------------------------------------
        # calc VAR surrogate
        if opt.var:
            if opt.multi:
                net = models.MultivariateVARNetwork()
                net.init(x, lags=opt.lag)
                y = surrogate.multivariate_var(x, net, surr_num=opt.surrnum, dist=opt.noise)
                save_result_files(opt, y, savename+'_var_multi')

        # calc Random Gaussian surrogate
        if opt.rg:
            if opt.multi:
                y = surrogate.multivariate_random_gaussian(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename+'_rg_multi')
            if opt.uni:
                y = surrogate.random_gaussian(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename+'_rg_uni')

        # calc Random Shuffling surrogate
        if opt.rs:
            if opt.multi:
                y = surrogate.multivariate_random_shuffling(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_rs_multi')
            if opt.uni:
                y = surrogate.random_shuffling(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_rs_uni')

        # calc Fourier Transform surrogate
        if opt.ft:
            if opt.multi:
                y = surrogate.multivariate_phase_randomized_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_ft_multi')
            if opt.uni:
                y = surrogate.phase_randomized_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_ft_uni')

        # calc AAFT surrogate
        if opt.aaft:
            if opt.multi:
                y = surrogate.multivariate_amplitude_adjusted_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_aaft_multi')
            if opt.uni:
                y = surrogate.amplitude_adjusted_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_aaft_uni')

        # calc AAFT surrogate
        if opt.iaaft:
            if opt.multi:
                y = surrogate.multivariate_iterated_amplitude_adjusted_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_iaaft_multi')
            if opt.uni:
                y = surrogate.iterated_amplitude_adjusted_ft(x, surr_num=opt.surrnum)
                save_result_files(opt, y, savename + '_iaaft_uni')

    plt.pause(1)
