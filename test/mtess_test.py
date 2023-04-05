# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import datetime

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import measures
import surrogate
import models


class TestMTESS(object):
    def __init__(self):
        self.f_name = 'cx-8x500-idx6.mat'
        self.work_path = 'data'

    def test(self):
        data_file = os.path.join(self.work_path, self.f_name)
        fdata = sio.loadmat(data_file)
        cx = fdata["CX"]
        cx = np.reshape(cx, (cx.shape[1],))  # to array
        node_num = cx[0].shape[0]
        sig_len = cx[0].shape[1]

        # check matlab compatibility
        '''
        mts, mtsp, nmts, nmtsp, means, stds, acs, cms, pcms, ccms, pccms = \
            measures.mtess.calc(cx=cx, mtrange=[0, 1], n_dft=100, cc_lags=8, pcc_lags=8)
#        measures.mtess.plot_all_mat(mts, mtsp, self.f_name)
#        measures.mtess.plot_bar3d(mts, self.f_name)
#        measures.mtess.plot_radar(mtsp[0, 1:7, :], self.f_name)
        '''
        # test surrogate algorithms
        ''
        dnames = ['original', 'mRG', 'mRS', 'mFT', 'mAAFT', 'VAR4', 'FT']
#        dnames = ['original', '2']
        cx2 = []
        cx2.append(cx[0])
        cx2.append(surrogate.multivariate_random_gaussian(cx[0])[:, :, 0])
        cx2.append(surrogate.multivariate_random_shuffling(cx[0])[:, :, 0])
        cx2.append(surrogate.multivariate_phase_randomized_ft(cx[0])[:, :, 0])
        cx2.append(surrogate.multivariate_amplitude_adjusted_ft(cx[0])[:, :, 0])
        net = models.MultivariateVARNetwork()
        net.init(cx[0], lags=4)
        cx2.append(surrogate.multivariate_var(cx[0], net)[:, :, 0])
        cx2.append(surrogate.phase_randomized_ft(cx[0])[:, :, 0])  # this one shows unstable result
        ret = measures.mtess.calc(cx=cx2, mtrange=[0, 1], n_dft=100, cc_lags=8, pcc_lags=8)
        measures.mtess.plot_radar(ret[1][0, 0:7, :], self.f_name, dnames=dnames)
        ''
        # synthetic lines
        dnames = ['original', 'flat@0.5', 'flat@0.9', 'random', 'sin']
#        dnames = ['original', '2']
        cx2 = []
        cx2.append(cx[0])
        cx2.append(np.ones((node_num, sig_len)) * 0.5)
        cx2.append(np.ones((node_num, sig_len)) * 0.9)
        cx2.append(np.random.rand(node_num, sig_len))
        y = np.sin(np.arange(sig_len) * np.pi / 8) * 0.5 + 0.5
        cx2.append(np.repeat(y.reshape((1, sig_len)), node_num, axis=0))
        ret = measures.mtess.calc(cx=cx2, mtrange=[0, 1], n_dft=100, cc_lags=8, pcc_lags=8)
        measures.mtess.plot_radar(ret[1][0, 0:5, :], self.f_name, dnames=dnames)
#        measures.mtess.plot_radar(ret[1][0, 0:2, :], self.f_name, dnames=dnames)
        y = 0


if __name__ == '__main__':
    print('start MTESS test')
    start_time = datetime.datetime.now()
    test_mt = TestMTESS()
    test_mt.test()
    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    print('run time: %d seconds' % int(interval))
