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
    'numeric',  # n-dimensional values. (e,g. MFCC)
    'events',  # detecting events (e.g, onsets)
    'hybrid',  # events with a value (e.g, peaks with amplitude)
    'bool',  # True/False test (e.g, presence of voice)
    'semantic',  # high level descriptor (e.g, key)
]

default_audio_types = ('wav', 'mp3', 'flac', 'ogg')  # To be extended


class QaTest:
    wrappers = dict()
    data = dict()
    metrics = dict()

    solutions = dict()
    ground_true = dict()
    times = dict()
    scores = dict()

    fs = 44100  # global fs for the test
    test_type = ''  # todo: restrict the possible values
    time_wrappers = True
    verbose = True

    def __init__(self, test_type='numeric', wrappers=[], metrics=[], time_wrappers=True, verbose=False):
        """

        :param test_type: {value, values, events, hybrid, bool}
        :param wrappers: list of QaWrapper objects to provide solutions
        :param metrics: list of QaMetric objects to assess the solutions
        :param time_wrappers: If true the Wrappers are timed.
        """

        self.verbose = verbose
        self.set_test_type(test_type)
        self.set_wrappers(wrappers)
        self.set_metrics(metrics)
        self.time_wrappers = time_wrappers

    def set_test_type(self, test_type):
        self.assert_test_type(test_type)
        self.test_type = test_type
        if self.verbose:
            print 'test_type is ok'

    def set_wrappers(self, wrappers):
        self.assert_wrappers(wrappers)
        for wrapper in wrappers:
            self.wrappers[wrapper.__class__.__name__] = wrapper
        if self.verbose:
            print 'Wrappers seem ok'

    def load_audio(self, filename, pattern=default_audio_types):
        """
        Uses Essentia MonoLoader to load the audio data. If filename is a folder it will look for all the `.extension` \
        files in the folder.
        :param filename:
        :param pattern:
        :return:
        """
        if not os.path.isfile(filename):
            files = [x for x in find_files(filename, pattern)]
        else:
            files = [filename]
        if len(files) == 0:
            raise EssentiaException("The file does not exist or the folder doesn't contain any '{}' file".format(pattern))
        try:
            for f in files:
                name = ''.join(os.path.basename(f).split('.')[:-1])
                self.data[name] = MonoLoader(filename=f, sampleRate=self.fs)()
        except IOError:
            if self.verbose:
                print 'cannot open {}'.format(f)  # todo improve exception handling
        # else:
        #    raise EssentiaException

    def set_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        for metric in metrics:
            self.metrics[metric.__class__.__name__] = metric
        if self.verbose:
            print 'metrics seem ok'

    def add_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        self.metrics.add(metrics)
        if self.verbose:
            print 'metrics seem ok'

    def load_solution(self, filename, name='', ground_true=False):
        if not os.path.isfile(filename):
            files = [x for x in find_files(filename, '')]
        else:
            files = [filename]

        for f in files:
            try:
                instance = ''.join(os.path.basename(f).split('.')[:-1])
                ext = (f.split('.')[-1])

                # Exclude known audio types so audio everything can be in the same folder
                if ext in default_audio_types:
                    continue

                parsers = {
                    'svl': lambda x: self.load_svl(x),
                    'csv': lambda x: self.load_csv(x),
                    'lab': lambda x: self.load_lab(x)
                    }

                try:
                    points = parsers[ext](f)
                except KeyError:
                    if self.verbose:
                        print ("ERROR: {} not loaded. \n"
                               "'{}' extension is not supported yet. Try with one of the supperted formats {}"
                               .format(f, ext, parsers.keys()))
                    continue

                if ground_true:
                    self.ground_true[instance] = np.array(points)
                else:
                    if name == '':
                        raise EssentiaException('If this solution is not the ground true it should be given a name')
                    self.solutions[name, instance] = points
            except EssentiaException:
                continue

    def load_svl(self):
        print 'This menthod is not implemeted in this class. Override it'
        return  # How should a general implementation be?

    def load_lab(self):
        print 'This menthod is not implemeted in this class. Override it'
        return  # How should a general implementation be?

    def compute(self, key_wrap, wrapper, key_inst, instance):
        if self.verbose:
            print "Computing file '{}' with the wrapper '{}'...".format(key_inst, key_wrap)
        self.solutions[key_wrap, key_inst] = wrapper.compute(instance)

    def compute_and_time(self, key_wrap, wrapper, key_inst, instance):
        if self.verbose:
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

    def compare_elapsed_times(self, output_file='stats.log'):
        w_names = self.wrappers.keys()
        i_names = self.data.keys()

        arrs = np.array([np.array([self.times[w, i] for i in i_names]) for w in w_names])
        means = np.mean(arrs, 1)

        fastest_idx = np.argmin(means)
        text = []
        text.append('')
        text.append('*' * 70)
        text.append('{} is the fastest method. Lasted {:.3f}s on average'
                    .format(w_names[fastest_idx], means[fastest_idx]))

        for i in range(len(w_names)):
            if i != fastest_idx:
                text.append('{} lasted {:.3f}s. {:.2f}x slower'
                            .format(w_names[i], means[i], means[i] / means[fastest_idx]))
        text.append('*' * 70)
        text.append('')

        if self.verbose:
            print '\n'.join(text)

        with open(output_file, 'a') as o_file:
            o_file.write('\n'.join(text))

    def score(self, key_wrap, key_inst, solution, key_metric, metric, gt):
        if self.verbose:
            print "Scoring file '{}' computed with the wrapper '{}' using metric '{}'..." \
                .format(key_inst, key_wrap, key_metric)
        self.scores[key_wrap, key_inst, key_metric] = metric.score(gt, solution)

    def score_all(self):
        gt_flags = []
        gt_name = ''
        for key_wrap, wrapper in self.wrappers.iteritems():
            gt_flags.append(wrapper.ground_true)
            if wrapper.ground_true:
                gt_name = wrapper.name

        if not bool(self.ground_true):

            #  assert only one GT
            if sum(gt_flags) > 1:
                raise EssentiaException("Only one wrapper can be set as ground true")
            if sum(gt_flags) == 0:
                raise EssentiaException("No wrapper is set as ground true")
        else:
            if sum(gt_flags) != 0:
                if self.verbose:
                    print("warning: When ground truth is set externally the ground truth wrapper would be ignored")

        for key_sol, solution in self.solutions.iteritems():
            key_wrap, key_inst = key_sol
            gt = self.ground_true.get(key_inst, None)
            if gt is not None:
                for key_metric, metric in self.metrics.iteritems():
                    self.score(key_wrap, key_inst, solution, key_metric, metric, gt)
                continue

            gt = self.solutions.get((gt_name, key_inst), None)
            if gt is not None:
                if key_wrap != gt_name:
                    for key_metric, metric in self.metrics.iteritems():
                        self.score(key_wrap, key_inst, solution, key_metric, metric, gt)
                    continue
            if self.verbose:
                print "{} was not scored because there is not ground truth available".format(key_inst)

    def save_test(self, output_file):
        import pickle
        with open('{}.pkl'.format(output_file), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_test(input_file):
        import pickle
        with open('{}.pkl'.format(input_file), 'rb') as f:
            return pickle.load(f)

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
