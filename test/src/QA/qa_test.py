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


import os
import numpy as np

from essentia import *
from essentia.utils import *
from essentia.standard import *
from essentia import array as esarr

EssentiaException = RuntimeError


test_types = [
    'values',  # n-dimensional values. (e,g. MFCC)
    'events',  # detecting events (e.g, onsets)
    'hybrid',  # events with a value (e.g, peaks with amplitude)
    'bool',  # True/False test (e.g, presence of voice)
    'semantic',  # high level descriptor (e.g, key)
]

default_audio_types = ('wav', 'mp3', 'flac', 'ogg')  # To be extended.


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename.lower().endswith(pattern):
                filename = os.path.join(root, basename)
                yield filename


class QaTest:
    wrappers = dict()
    data = dict()
    routes = dict()
    metrics = dict()

    solutions = dict()
    ground_true = dict()
    times = dict()
    scores = dict()

    fs = 44100.  # Global fs for the test.
    test_type = ''  # TODO: Restrict the possible values.
    time_wrappers = True
    verbose = True
    log = True

    def __init__(self, test_type='values', wrappers=[], metrics=[], time_wrappers=True, verbose=False, log=True):
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
        self.log = log

    def set_test_type(self, test_type):
        self.assert_test_type(test_type)
        self.test_type = test_type
        if self.verbose:
            print('test_type is ok')

    def set_wrappers(self, wrappers):
        self.assert_wrappers(wrappers)
        for wrapper in wrappers:
            self.wrappers[wrapper.__class__.__name__] = wrapper
        if self.verbose:
            print('Wrappers seem ok')

    def load_audio(self, filename, pattern=default_audio_types, stereo=False):
        """
        Uses Essentia MonoLoader to load the audio data. If filename is a folder it will look for all the `.extension` \
        files in the folder.
        """
        if not os.path.isfile(filename):
            files = [x for x in find_files(filename, pattern)]
        else:
            files = [filename]
        if len(files) == 0:
            raise EssentiaException("The file does not exist or the folder doesn't contain any '{}' file".format(pattern))
        try:
            for f in files:
                name = '.'.join(os.path.basename(f).split('.')[:-1])
                if stereo:
                    audio, fs, _, _, _, _ = AudioLoader(filename=f)()
                    if np.abs(fs - self.fs) > 1e-3:
                        resampler = Resample(inputSampleRate=fs, outputSampleRate=self.fs)
                        l = resampler(audio[:,0])
                        r = resampler(audio[:,1])
                        audio = np.vstack([l, r])
                    self.data[name] = audio
                else:
                    self.data[name] = MonoLoader(filename=f, sampleRate=self.fs)()
                self.routes[name] = f
        except IOError:
            if self.verbose:
                print('cannot open {}'.format(f))  # TODO improve exception handling.
        # else:
        #    raise EssentiaException

    def set_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        for metric in metrics:
            self.metrics[metric.__class__.__name__] = metric
        if self.verbose:
            print('Metrics seem ok.')

    def add_metrics(self, metrics):
        if not type(metrics) == list:
            metrics = [metrics]
        self.assert_metrics(metrics)
        self.metrics.add(metrics)
        if self.verbose:
            print('Metrics seem ok.')

    def load_solution(self, filename, name='', ground_true=False):
        if not os.path.isfile(filename):
            files = [x for x in find_files(filename, '')]
        else:
            files = [filename]

        for f in files:
            try:
                instance = '.'.join(os.path.basename(f).split('.')[:-1])
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
                        print("ERROR: {} not loaded. \n"
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
        print('This menthod is not implemeted in this class. Override it')
        return  # How should a general implementation be?

    def load_lab(self):
        print('This menthod is not implemeted in this class. Override it')
        return  # How should a general implementation be?

    def compute(self, key_wrap, wrapper, key_inst, instance):
        if self.verbose:
            print("Computing file '{}' with the wrapper '{}'...".format(key_inst, key_wrap))
        self.solutions[key_wrap, key_inst] = wrapper.compute(self, instance, key_inst, key_wrap)

    def compute_and_time(self, key_wrap, wrapper, key_inst, instance):
        if self.verbose:
            print("Computing file '{}' with the wrapper '{}'...".format(key_inst, key_wrap))
        solution, time = wrapper.compute_and_time(self, instance, key_inst, key_wrap)
        self.solutions[key_wrap, key_inst] = solution
        self.times[key_wrap, key_inst] = time

    def compute_all(self, output_file='compute.log'):
        for key_wrap, wrapper in self.wrappers.items():
            for key_inst, instance in self.data.items():
                if self.time_wrappers:
                    self.compute_and_time(key_wrap, wrapper, key_inst, instance)
                else:
                    self.compute(key_wrap, wrapper, key_inst, instance)

        if self.log:
            np.set_printoptions(precision=3)
            final_text = []
            for data_key in self.data.keys():
                text = ['Results for: {}\n'.format(data_key)]
                found_something = False
                for wrapper_key in self.wrappers.keys():
                    if self.solutions[(wrapper_key, data_key)].any():
                        found_something = True
                        text.append('Using {}:'.format(wrapper_key))
                        text.append(str(self.solutions[(wrapper_key, data_key)]))
                        text.append('\n')
                if found_something:
                    final_text.append(''.join(text))
            with open(output_file, 'w') as o_file:
                o_file.write(''.join(final_text))
            if self.time_wrappers:
                self.compare_elapsed_times(output_file=output_file, mode='a')

    def compare_elapsed_times(self, output_file='stats.log', mode='a'):
        w_names = list(self.wrappers.keys())
        i_names = list(self.data.keys())

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
            print('\n'.join(text))

        with open(output_file, mode) as o_file:
            o_file.write('\n'.join(text))

    def score(self, key_wrap, key_inst, solution, key_metric, metric, gt):
        if self.verbose:
            print("Scoring file '{}' computed with the wrapper '{}' using metric '{}'..."
                  .format(key_inst, key_wrap, key_metric))
        self.scores[key_wrap, key_inst, key_metric] = metric.score(gt, solution)

    def score_all(self):
        gt_flags = []
        gt_name = ''
        for key_wrap, wrapper in self.wrappers.items():
            gt_flags.append(wrapper.ground_true)
            if wrapper.ground_true:
                gt_name = wrapper.name

        if not bool(self.ground_true):

            #  Assert only one GT.
            if sum(gt_flags) > 1:
                raise EssentiaException("Only one wrapper can be set as ground true.")
            if sum(gt_flags) == 0:
                raise EssentiaException("No wrapper is set as ground true.")
        else:
            if sum(gt_flags) != 0:
                if self.verbose:
                    print("warning: When ground truth is set externally the ground truth wrapper would be ignored.")

        for key_sol, solution in self.solutions.items():
            key_wrap, key_inst = key_sol
            gt = self.ground_true.get(key_inst, None)
            if gt is not None:
                for key_metric, metric in self.metrics.items():
                    self.score(key_wrap, key_inst, solution, key_metric, metric, gt)
                continue

            gt = self.solutions.get((gt_name, key_inst), None)
            if gt is not None:
                if key_wrap != gt_name:
                    for key_metric, metric in self.metrics.iteritems():
                        self.score(key_wrap, key_inst, solution, key_metric, metric, gt)
                    continue
            if self.verbose:
                print("{} was not scored because there is not ground truth available.".format(key_inst))

    def save_test(self, output_file):
        import pickle
        with open('{}.pkl'.format(output_file), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def remove_wrapper(self, name):
        if name in self.wrappers:
            del self.wrappers[name]
        else:
            print('No wrapper named {} found.'.format(name))

    def clear_wrappers(self):
        self.wrappers.clear()

    def remove_solution(self, name):
        if name in self.solutions:
            del self.solutions[name]
        else:
            print('No solution named {} found.'.format(name))

    def clear_solutions(self, wrapper=''):
        if wrapper is '':
            self.solutions.clear()
        else:
            if wrapper in self.wrappers:
                names = self.filter(self.solutions.keys(), 0, wrapper)
                for name in names:
                    self.remove_solution(name)
            else:
                print('No wrapper named {} found.'.format(wrapper))

    def remove_scores(self, name):
        if name in self.scores:
            del self.solutions[name]
        else:
            print('No score named {} found.'.format(name))

    def clear_scores(self, wrapper='', metric=''):
        if wrapper is '' and metric is '':
            self.solutions.clear()
        else:
            names = self.solutions.keys()
            if wrapper in self.wrappers:
                names = self.filter(names, 0, wrapper)

            if metric in self.metrics:
                names = self.filter(names, 2, metric)

            for name in names:
                self.remove_solution(name)

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
                raise EssentiaException('Input should be wrappers.')
            types.add(wrapper.test_type)
        if len(types) > 1:
            raise EssentiaException("Wrappers type don't match.")

    @staticmethod
    def assert_test_type(test_type):
        if test_type.lower() not in test_types:
            raise EssentiaException("'{}' is not supported among the test types ({})."
                                    .format(test_type, test_types))

    @staticmethod
    def assert_metrics(metrics):
        for metric in metrics:
            if not isinstance(metric, QaMetric):
                raise EssentiaException('Metrics should be wrapped in a QaMetric object.')

    @staticmethod
    def filter(tuples, pos, patt):
        aux = [idx for idx, x in enumerate(tuples) if x[pos] == patt]
        filtered = []

        for idx in aux:
            filtered.append(tuples[idx])
        return filtered


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
        import timeit
        import functools

        t = timeit.Timer(functools.partial(self.compute, *args))
        secs = t.timeit(1)

        solution = self.compute(*args)

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
