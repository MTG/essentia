#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/


from qa_test import *
from qa_testevents import QaTestEvents
from qa_testvalues import QaTestValues
import essentia.standard as es

import matplotlib.pyplot as plt
from essentia import instantPower
from essentia import db2pow
from scipy.special import iv
from scipy.constants import pi
from math import isnan
from math import isinf
frameSize = 512
hopSize = frameSize / 2
noiseThreshold = -40
eps = (np.finfo(np.float32).eps)

class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.SNR(frameSize=frameSize, noiseThreshold=noiseThreshold)

    def compute(self, *args):
        self.algo.reset()
        for frame in es.FrameGenerator(args[1], frameSize=frameSize, hopSize=hopSize,
                                       startFromZero=True):
            _, _, snr = self.algo(frame)

        return esarr([snr])


class Dev(QaWrapper):
    """
    Development Solution.
    """

    def compute(self, *args):
        eps = (np.finfo(np.float32).eps)
        def SNR_prior_est(alpha, mmse, noise_pow, snr_inst):
            return alpha * (np.abs(mmse) ** 2) / noise_pow + (1 - alpha) *\
                   np.clip(snr_inst, a_min=0, a_max=None)

        def update_noise_psd(noise_spectrum, noise, alpha=.98):
            return alpha * noise_spectrum + (1 - alpha) * np.abs(noise) ** 2

        def update_noise_pow(noise_std, noise, alpha=.98):
            return alpha * noise_std + (1 - alpha) * noise

        def update_y(mean_y, y, alpha=.98):
            return alpha * mean_y + (1 - alpha) * y

        def MMSE(v, snr_post, Y):
            g = 0.8862269254527579 # gamma(1.5)

            output = np.zeros(len(v))

            for idx in range(len(Y)):
                if v[idx] > 10:
                    output[idx] = v[idx] * Y[idx] / snr_post[idx]
                else:
                    output[idx] = g * ( np.sqrt(v[idx]) / (snr_post[idx] + eps)) *\
                                  np.exp(-v[idx] / 2.) *\
                                  ((1 + v[idx]) * iv(0., v[idx] / 2.) +\
                                  v[idx] * iv(1., v[idx] / 2.)) * Y[idx]
            return output

        def SNR_post_est(Y, noise_pow):
            return np.abs(Y) ** 2 / noise_pow

        def SNR_inst_est(snr_post_est):
            return snr_post_est - 1.

        def V(snr_prior, snr_post):
            return (snr_prior / (1. + snr_prior)) * snr_post


        x = esarr(args[1])
        asume_gauss_psd = args[2]
        idx_ = 0

        silenceThreshold = db2pow(noiseThreshold)

        MMSE_alpha = .98
        noise_alpha = .9
        snr_alpha = .95

        y = []

        noise_psd = np.zeros(frameSize / 2 + 1, dtype=np.float32)

        previous_snr_prior = np.zeros(frameSize / 2 + 1, dtype=np.float32)
        previous_snr_inst = np.zeros(frameSize / 2 + 1, dtype=np.float32)
        previous_snr_post = np.zeros(frameSize / 2 + 1, dtype=np.float32)
        previous_Y = np.zeros(frameSize / 2 + 1, dtype=np.float32)
        previous_noise_psd = np.zeros(frameSize / 2 + 1, dtype=np.float32)

        noise_std = 0
        ma_snr_average = 0

        spectrum = es.Spectrum(size=frameSize)
        window = es.Windowing(size=frameSize, type='hann', normalized=False)

        for frame in es.FrameGenerator(x, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
            Y = spectrum(window(frame))

            if instantPower(frame) < silenceThreshold:
                # Fixed threshold
                # noise_std = 1e-5

                # Estimated threshold
                # noise_std = update_noise_pow(noise_std, instantPower(frame), alpha=alpha)
                # noise_spectrum = np.ones(frameSize / 2 + 1) * noise_std

                noise_psd = update_noise_psd(noise_psd, Y, alpha=noise_alpha)

                noise_std = np.mean(noise_psd)

                snr_post = SNR_post_est(Y, noise_psd)
                snr_inst = SNR_inst_est(snr_post)

            else:
                if np.sum(previous_snr_prior) == 0:
                    previous_snr_prior =  MMSE_alpha + (1 - MMSE_alpha) * np.clip(previous_snr_inst, a_min=0., a_max=None)

                    if 0:
                        noise_psd = np.ones(frameSize / 2 + 1) *np.mean(noise_psd)

                snr_post = SNR_post_est(Y, noise_psd)
                snr_inst = SNR_inst_est(snr_post)

                v = V(previous_snr_prior, previous_snr_post)

                previous_mmse = MMSE(v, previous_snr_post, previous_Y)

                snr_prior = SNR_prior_est(MMSE_alpha, previous_mmse, previous_noise_psd, snr_inst)

                X_psd_est = noise_psd * snr_prior

                snr_average = np.mean(X_psd_est) / np.mean(noise_psd)


                ma_snr_average = update_y(ma_snr_average, snr_average, alpha=snr_alpha)

                previous_snr_prior = snr_prior

            previous_noise_psd = noise_psd
            previous_snr_post = snr_post
            previous_snr_inst = snr_inst
            previous_Y = Y

            idx_ += 1

        return esarr([ma_snr_average])


if __name__ == '__main__':
    folder = 'SNR'

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('events')
        # Dev('events')
    ]

    # Instantiating the test
    qa = QaTestValues(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/Vivaldi_Sonata_5_II_Allegro_prominent_jump.wav'
    # data_dir = '../../QA-audio/Clicks'
    # data_dir = '/home/pablo/reps/essentia/test/audio/recorded/vignesh.wav'
    data_dir = '/home/pablo/reps/essentia/test/audio/recorded/'
    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/'
    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/vignesh_prominent_jump.wav'
    # data_dir = '../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/'

    qa.load_audio(filename=data_dir, stereo=False)  # Works for a single
    # qa.load_solution(data_dir, ground_true=True)

    fs = 44100.
    time = 5 #s



    # noise_durations = [1.] #s
    # time_axis = np.arange(0, time, 1 / fs)
    #
    # nsamples = len(time_axis)
    # for noise_alpha in [.9]:
    #     for asume_gauss_psd in [0]:
    #         for noise_only in noise_durations:
    #
    #             results = []
    #             gt = []
    #             for i in range(1):
    #                 noise = np.random.randn(nsamples)
    #                 noise /= np.std(noise)
    #
    #
    #                 signal = np.sin(2 * pi * 5000 * time_axis)
    #
    #                 signal_db = -22.
    #                 noise_db  = -50.
    #
    #                 noise_var = instantPower(esarr(db2amp(noise_db) * noise))
    #                 signal[:int(noise_only * fs)] = np.zeros(int(noise_only * fs))
    #                 real_snr_prior = 10. * np.log10(
    #                     (instantPower(esarr(db2amp(signal_db) * signal[int(noise_only * fs):]))) /
    #                     (instantPower(esarr(db2amp(noise_db)  * noise[int(noise_only * fs):]))))
    #
    #                 real_snr_prior_esp_corrected = real_snr_prior - 10. * np.log10(fs / 2.)
    #                 gt.append(real_snr_prior_esp_corrected)
    #
    #                 signal_and_noise = esarr(db2amp(signal_db) * signal + db2amp(noise_db) * noise)
    #
    #                 ma_snr_average = qa.wrappers['Dev'].compute(None, signal_and_noise, asume_gauss_psd, noise_alpha)
    #                 mean_snr_estimation = 10 * np.log10(ma_snr_average)
    #                 mean_snr_estimation_corrected = mean_snr_estimation - 10. * np.log10(fs / 2.)
    #                 print 'with dev, error: {:.3f}dB'.format(np.abs(mean_snr_estimation_corrected[0] - real_snr_prior_esp_corrected))
    #
    #                 ma_snr_average = qa.wrappers['EssentiaWrap'].compute(None, signal_and_noise, asume_gauss_psd, noise_alpha)
    #                 mean_snr_estimation = 10 * np.log10(ma_snr_average)
    #                 mean_snr_estimation_corrected = mean_snr_estimation - 10. * np.log10(fs / 2.)
    #                 print 'with Esssentia, error: {:.3f}dB'.format(np.abs(mean_snr_estimation_corrected[0] - real_snr_prior_esp_corrected))
    #
    #
    #                 results.append(mean_snr_estimation_corrected)
                # print '*' * 30
                # print 'Noise duration is {:.3f}.'.format(noise_only)
                # print 'Noise alpha is {:.2f}.'.format(noise_alpha)
                # print 'SNR estimation mean is {:.4f} with a std of {:.3f}.'.format(np.mean(results), np.std(results))
                # print 'SNR groud true mean is {:.4f} with a std of {:.3f}.'.format(np.mean(gt), np.std(gt))
                # if asume_gauss_psd:
                #     print 'assuming gaussin psd for noise (flat psd)'
                # else:
                #     print 'estimating noise psd shape from data'
                # print '*' * 30


    # Compute and the results, the scores and and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))

    # qa.score_all()
    # qa.scores
    # qa.save_test('{}/test'.format(folder))
