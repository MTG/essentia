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
import numpy as np
from librosa.effects import trim
import essentia.standard as es
from essentia import array as esarray
from scipy.signal import medfilt

order = 3
frame_size = 512
hop_size = 256
kernel_size = 7
times_thld = 8
energy_thld = 0.001
sub_frame = 32

class DevWrap(QaWrapper):
    """
    Essentia Solution.
    """
    errors = []
    errors_filt = []
    samples_peaking_frame = []
    frame_idx = []
    frames = []
    power = []

    def compute(self, x):

        LPC = es.LPC(order=order, type='regular')
        W = es.Windowing(size=frame_size, zeroPhase=False, type='triangular')
        predicted = np.zeros(hop_size)
        y = []
        self.frames = []
        self.errors = []
        self.errors_filt = []
        self.samples_peaking_frame = []
        self.frame_idx = []
        self.power = []
        frame_counter = 0
        from time import clock
        lpc_timer = 0
        pre_timer = 0
        med_timer = 0
        mas_timer = 0

        for frame in es.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
            self.power.append(es.essentia.instantPower(frame))
            self.frames.append(frame)
            frame_un = np.array(frame[hop_size / 2: hop_size * 3 / 2])
            frame = W(frame)
            norm = np.max(np.abs(frame))
            if not norm:
                continue
            frame /= norm

            c1 = clock()
            lpc_f, _ = LPC(esarray(frame))

            lpc_timer += (clock() - c1)

            c1 = clock()
            lpc_f1 = lpc_f[1:][::-1]

            for idx, i in enumerate(range(hop_size / 2, hop_size * 3 / 2)):
                predicted[idx] = - np.sum(np.multiply(frame[i - order:i], lpc_f1))
            pre_timer += (clock() - c1)

            error = np.abs(frame[hop_size/2: hop_size * 3 / 2] - predicted)

            threshold1 = times_thld * np.std(error)

            c1 = clock()
            med_filter = medfilt(error, kernel_size=kernel_size)
            filtered = np.abs(med_filter - error)
            med_timer += (clock() - c1)

            c1 = clock()
            mask = []
            for i in range(0, len(error), sub_frame):
                # r = np.sum(error[i:i + sub_frame]) / float(sub_frame) > (np.median(error))
                # r = np.sum(filtered[i:i + sub_frame]) / float(sub_frame) > error_thld
                # r = np.sum(filtered[i:i + sub_frame]) / float(sub_frame) > (np.median(filtered) )
                r = es.essentia.instantPower(frame_un[i:i + sub_frame]) > energy_thld
                mask += [r] * sub_frame
            mask = mask[:len(error)]
            mask = np.array([mask]).astype(float)[0]
            mas_timer += (clock() - c1)

            if sum(mask) == 0:
                threshold2 = 1000  # just skip silent frames
            else:
                threshold2 = times_thld * (np.std(error[mask.astype(bool)]) +
                                           np.median(error[mask.astype(bool)]))

            threshold = np.max([threshold1, threshold2])

            samples_peaking = np.sum(filtered >= threshold)
            if samples_peaking >= 1:
                y.append(frame_counter * hop_size / 44100.)
                self.frame_idx.append(frame_counter)

            self.frames.append(frame)
            self.errors.append(error)
            self.errors_filt.append(filtered)
            self.samples_peaking_frame.append(samples_peaking)

            frame_counter += 1

        """
        print 'computing lpcs: {:.2f}s'.format(lpc_timer)
        print 'making predictions: {:.2f}s'.format(pre_timer)
        print 'computing median filter: {:.2f}s'.format(med_timer)
        print 'computing mask: {:.2f}s'.format(mas_timer)
        print '*' * 20
        """
        return np.array(y)


if __name__ == '__main__':

    # Instantiating wrappers
    wrappers = [
        DevWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../QA-audio/Jumps/random_jumps'

    # Add the testing files
    # qa.set_data(filename='../../QA-audio/Jumps/prominent_jumps')  # Works for a single
    # qa.set_data(filename='../../../../../data/Dead_Combo_-_01_-_Povo_Que_Cas_Descalo_silence.wav')  # Works for a single
    # qa.set_data(filename='../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/03. Desakato - Estigma.mp3')  # Works for a single
    #  qa.load_audio(filename='../../QA-audio/Jumps/loud_songs/')  # Works for a single

    qa.load_audio(filename=data_dir)  # Works for a single
    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times

    qa.compute_all()

    qa.score_all()

    precision = []
    recall = []
    f_measure = []
    for i in qa.scores.itervalues():
        precision.append(i['Precision'])
        recall.append(i['Recall'])
        f_measure.append(i['F-measure'])

    print 'Mean Precision: {}'.format(np.mean(precision))
    print 'Mean Recall: {}'.format(np.mean(recall))
    print 'Mean F-measure: {}'.format(np.mean(f_measure))

    """
    # Add extra metrics
    qa.set_metrics(Distance())

    # Add ground true
    qa.load('../../QA-audio/StartStopSilence/', ground_true=True)

    qa.plot_all(force=True, plots_dir='StartStopSilence/plots')

    qa.score_all()

    qa.generate_stats(output_file='StartStopSilence/stats.log')

    qa.compare_elapsed_times(output_file='StartStopSilence/stats.log')

    qa.save_test('StartStopSilence/test')
    """