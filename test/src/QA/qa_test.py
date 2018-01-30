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



from essentia import *
from essentia.utils import *
from essentia.standard import *
import numpy as np
import os

EssentiaException = RuntimeError


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename.lower().endswith(pattern):
                filename = os.path.join(root, basename)
                yield filename


test_types = [
    'values',    # several float values. (e,g. MFCC)
    'value',     # only one value. (e.g, average loudness)
    'events',    # detecting events (e.g, onsets)
    'hybrid',    # events with a value (e.g, peaks with amplitude)
    'bool',      # True/False test (e.g, presence of voice)
    'semantic',  # high level descriptor (e.g, key)
]


class QaTest:
    wrappers = dict()
    data = dict()
    metrics = dict()

    solutions = dict()
    times = dict()
    scores = dict()

    test_type = ''  # todo: restrict the possible values
    time_wrappers = True
    verbose = True

    def __init__(self, test_type='values', wrappers=[], metrics=[], time_wrappers=True, verbose=True):
        """

        :param test_type: {value, values, events, hybrid, bool}
        :param wrappers: list of QaWrapper objects to provide solutions
        :param metrics: list of QaMetric objects to assess the solutions
        :param time_wrappers: If true the Wrappers are timed.
        """

        self.set_test_type(test_type)
        self.set_wrappers(wrappers)
        self.set_metrics(metrics)
        self.time_wrappers = time_wrappers
        self.verbose = verbose

    def set_test_type(self, test_type):
        self.assert_test_type(test_type)
        self.test_type = test_type
        print 'test_type is ok'

    def set_wrappers(self, wrappers):
        self.assert_wrappers(wrappers)
        for wrapper in wrappers:
            self.wrappers[wrapper.__class__.__name__] = wrapper
        print 'Wrappers seem ok'

    def set_data(self, filename, sample_rate=44100, pattern='.wav'):
        """
        Uses Essentia MonoLoader to load the audio data. If filename is a folder it will look for all the `.extension` \
        files in the folder.
        :param filename:
        :param sample_rate:
        :param pattern:
        :return:
        """
        if not os.path.isfile(filename):
            files = [x for x in find_files(filename, pattern)]
        else:
            files = [filename]

        try:
            for f in files:
                self.data[os.path.basename(f)] = MonoLoader(filename=f, sampleRate=sample_rate)()
        except IOError:
            print 'cannot open {}'.format(f)  # todo improve exception handling
        #else:
        #    raise EssentiaException

    def set_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        for metric in metrics:
            self.metrics[metric.__class__.__name__] = metric
        print 'metrics seem ok'

    def add_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        self.metrics.add(metrics)
        print 'metrics seem ok'

    def compute(self, key_wrap, wrapper, key_inst, instance):
        print "Computing file '{}' with the wrapper '{}'...".format(key_inst, key_wrap)
        self.solutions[key_wrap, key_inst] = wrapper.compute(instance)

    def compute_and_time(self, key_wrap, wrapper, key_inst, instance):
        print "Computing file '{}' with the wrapper '{}'...".format(key_inst, key_wrap)
        solution, time = wrapper.compute_and_time(instance)
        self.solutions[key_wrap, key_inst] = solution
        self.times[key_wrap, key_inst] = time

    def compute_all(self):
        for key_wrap, wrapper in self.wrappers.iteritems():
            for key_inst, instance in self.data.iteritems():
                if self.time_wrappers:
                    self.compute_and_time(key_wrap, wrapper, key_inst, instance)
                else:
                    self.compute(key_wrap, wrapper, key_inst, instance)

    def compare_elapsed_times(self):
        w_names = self.wrappers.keys()
        i_names = self.data.keys()

        arrs = np.array([np.array([self.times[w, i] for i in i_names]) for w in w_names])
        means = np.mean(arrs, 1)

        fastest_idx = np.argmin(means)
        print ''
        print '*' * 70
        print '{} is the fastest method. Lasted {:.3f}s on average'\
            .format(w_names[fastest_idx], means[fastest_idx])

        for i in range(len(w_names)):
            if i != fastest_idx:
                print '{} lasted {:.3f}s. {:.2f}x slower'\
                    .format(w_names[i], means[i], means[i] / means[fastest_idx])
        print '*' * 70
        print ''

    def score(self, key_wrap, key_inst, solution, key_metric, metric, gt):
        print "Scoring file '{}' computed with the wrapper '{}' using metric '{}'..."\
            .format(key_inst, key_wrap, key_metric)
        self.scores[key_wrap, key_inst, key_metric] = metric.score(gt, solution)

    def score_all(self):
        gt_flags = []
        for key_wrap, wrapper in self.wrappers.iteritems():
            gt_flags.append(wrapper.ground_true)
            if wrapper.ground_true:
                gt_name = wrapper.name
        #  assert only one GT
        if sum(gt_flags) > 1:
            raise EssentiaException("Only one wrapper can be set as ground true")
        if sum(gt_flags) == 0:
            raise EssentiaException("No wrapper is set as ground true")

        for key_sol, solution in self.solutions.iteritems():
            key_wrap, key_inst = key_sol
            gt = self.solutions[gt_name, key_inst]
            if key_wrap != gt_name:
                for key_metric, metric in self.metrics.iteritems():
                    self.score(key_wrap, key_inst, solution, key_metric, metric, gt)

    @staticmethod
    def assert_wrappers(wrappers):
        types = set()
        for wrapper in wrappers:
            if not isinstance(wrapper, QaWrapper):
                raise EssentiaException('Input should be wrappers')
            types.add(wrapper.test_type)
        if len(types) > 1:
            raise EssentiaException("Wrappers type don't match")

    @staticmethod
    def assert_test_type(test_type):
        if test_type.lower() not in test_types:
            raise EssentiaException("'{}' is not supported among the test types ({})"
                                    .format(test_type, test_types))

    @staticmethod
    def assert_metrics(metrics):
        for metric in metrics:
            if not isinstance(metric, QaMetric):
                raise EssentiaException('Metrics should be wrapped in a QaMetric object')

    def run(self):
        pass


class QaWrapper:
    test_type = ''
    ground_true = False
    name = ''

    def __init__(self, test_type='values', ground_true=False):
        QaTest.assert_test_type(test_type)
        self.test_type = test_type
        self.ground_true = ground_true
        self.name = self.__class__.__name__
        pass

    def compute(self):
        """
        This method should be overloaded in the sibling class.
        :return:
        """
        pass

    def compute_and_time(self, *args):
        from time import clock

        start = clock()
        solution = self.compute(*args)
        end = clock()

        secs = (end - start)

        return solution, secs

class QaMetric:
    def __init__(self):
        pass

    def score(self):
        """
        This method should be overloaded in the sibling class.
        :return:
        """
        pass
