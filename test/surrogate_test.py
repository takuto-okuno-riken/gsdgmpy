# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import datetime

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import surrogate
import measures
import models


class TestSurrogateRS(object):
    def __init__(self):
        self.f_name = 'ww32-1.mat'
        self.work_path = 'data'

    def plot_signals(self, x, name):
        plt.figure()
        plt.plot(np.transpose(x), linewidth=0.3)
        plt.title('time-series ' + name)
        plt.xlabel('Time frames')
        plt.show(block=False)

    def check_similarity(self, x, y, t_range, n_dft, prefix):
        node_num = x.shape[0]
        z = np.stack([x, y], axis=2)

        e = np.eye(node_num)
        nanx = np.where(e == 1, np.nan, 0)

        # calc mean difference
        sm = np.mean(z, axis=1)
        dm = sm[:, 0] - sm[:, 1]
        msim = np.abs(dm) / (t_range / 2)

        # calc std dev difference
        ss = np.std(z, axis=1)
        ds = ss[:, 0] - ss[:, 1]
        ssim = np.abs(ds) / (t_range / 4)

        # calc amplitude difference
#        plt.figure()
#        d, p1 = measures.dft.plot(x=x, n_dft=n_dft)  # DFT OK with matlab ver.
        d, p1 = measures.dft.calc(x=z, n_dft=n_dft)
        asim = measures.cos_sim(d[:, :, 0], d[:, :, 1])

        # calc CM
#        plt.figure()
#        xfc = measures.cm.plot(x=x)  # FC OK with matlab ver.
#        xc = measures.cm.calc(x=x)
#        yc = measures.cm.calc(x=y)
#        fcsim = measures.cos_sim(xc, yc)

        # calc CCM
        xcc = measures.ccm.calc(x=x, max_lag=4)
        ycc = measures.ccm.calc(x=y, max_lag=4)
        xc2 = xcc[:, :, 4] + nanx
        yc2 = ycc[:, :, 4] + nanx
        fcsim = measures.cos_sim(xc2, yc2)
        xcc = np.delete(xcc, 4, axis=2)  # CC OK with matlab ver.
        ycc = np.delete(ycc, 4, axis=2)
        nx8 = np.repeat(nanx[:, :, np.newaxis], 8, axis=2)
        ccsim = measures.cos_sim(xcc + nx8, ycc + nx8)

        # calc PCM
#        plt.figure()
#        xpc = measures.pcm.plot(x=x)
#        xpc = measures.pcm.calc(x=x)
#        ypc = measures.pcm.calc(x=y)
#        pcsim = measures.cos_sim(xpc, ypc)

        # calc PCCM
        xpcc2 = measures.pccm_.calc(x=x, max_lag=2)
        xpcc = measures.pccm.calc(x=x, max_lag=2)
        print('xpcc2 diff='+str(np.sum(np.abs(xpcc2-xpcc))))

        ypcc = measures.pccm.calc(x=y, max_lag=2)
        xc2 = xpcc[:, :, 2] + nanx  # PC OK with matlab ver.
        yc2 = ypcc[:, :, 2] + nanx
        pcsim = measures.cos_sim(xc2, yc2)
        xpcc = np.delete(xpcc, 2, axis=2)  # PCC OK with matlab ver.
        ypcc = np.delete(ypcc, 2, axis=2)
        nx4 = np.repeat(nanx[:, :, np.newaxis], 4, axis=2)
        pccsim = measures.cos_sim(xpcc + nx4, ypcc + nx4)

        print(prefix+' : m='+format(1-np.mean(msim),'3f')+', s='+format(1-np.mean(ssim),'3f')+', ac='+format(asim,'3f')+
              ', fc='+format(fcsim,'3f')+', pc='+format(pcsim,'3f')+', ccm='+format(ccsim,'3f')+', pccm='+format(pccsim,'3f'))

    def test(self):
        data_file = os.path.join(self.work_path, self.f_name)
        fdata = sio.loadmat(data_file)
        node_num = 32   # node number
        sig_len = 300  # signal length
        si = np.array(fdata["X"])
        x = si[0:node_num, 0:sig_len]
        self.plot_signals(x, name='original')

        self.check_similarity(x=x, y=x, t_range=1, n_dft=100, prefix='same signal')

        y = surrogate.random_shuffling(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad RS surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='RS surrogate')

        y = surrogate.multivariate_random_shuffling(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad MRS surrogate')
        self.plot_signals(y[:, :, 2], name='MRS surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='MRS surrogate')

        y = surrogate.random_gaussian(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad RG surrogate')
        self.plot_signals(y[:, :, 2], name='RG surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='RG surrogate')

        y = surrogate.multivariate_random_gaussian(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad MRG surrogate')
        self.plot_signals(y[:, :, 2], name='MRG surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='MRG surrogate')

        y = surrogate.phase_randomized_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad FT surrogate')
        self.plot_signals(y[:, :, 2], name='FT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='FT surrogate')

        y = surrogate.multivariate_phase_randomized_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad MFT surrogate')
        self.plot_signals(y[:, :, 2], name='MFT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='MFT surrogate')

        y = surrogate.amplitude_adjusted_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad AAFT surrogate')
        self.plot_signals(y[:, :, 2], name='AAFT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='AAFT surrogate')

        y = surrogate.multivariate_amplitude_adjusted_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad MAAFT surrogate')
        self.plot_signals(y[:, :, 2], name='MAAFT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='MAAFT surrogate')

        y = surrogate.iterated_amplitude_adjusted_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad IAAFT surrogate')
        self.plot_signals(y[:, :, 2], name='IAAFT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='IAAFT surrogate')

        y = surrogate.multivariate_iterated_amplitude_adjusted_ft(x=x, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad MIAAFT surrogate')
        self.plot_signals(y[:, :, 2], name='MIAAFT surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='MIAAFT surrogate')

        net = models.MultivariateVARNetwork()
        net.init(x, lags=3)
        y = surrogate.multivariate_var(x, net, surr_num=10)
        if np.all(y[:, :, 2] == y[:, :, 3]):
            print('bad VAR(3) surrogate')
        self.check_similarity(x=x, y=y[:, :, 3], t_range=1, n_dft=100, prefix='VAR(3) surrogate')

        y = 0


if __name__ == '__main__':
    print('start RS surrogate test')
    start_time = datetime.datetime.now()
    test_rs = TestSurrogateRS()
    test_rs.test()
    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    print('run time: %d seconds' % int(interval))
