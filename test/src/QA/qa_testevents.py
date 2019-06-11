#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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


"""
Class for event tests. Here the methods can be overrided to fit the event case demands
"""


from qa_test import *


class QaTestEvents(QaTest):
    def __init__(self, *args, **kwargs):
        # Use QaTest constructor with hardcoded `test_type`.
        if len(args) > 1:
            args = args[1:]
        kwargs.pop('test_type', None)

        QaTest.__init__(self, 'events', *args, **kwargs)

        # Add mir_eval metrics by default when available.
        self.add_mir_eval()

    def load_svl(self, filename):
        import xml.etree.ElementTree as et
        tree = et.parse(filename)
        root = tree.getroot()
        sr = float(root.find('data').find('model').get('sampleRate'))
        points = []
        for point in root.find('data').find('dataset'):
            points.append(float(point.get('frame')) / sr)
        return points

    def load_lab(self, filename):
        points = []
        with open(filename, 'r') as i_file:
            lines = i_file.readlines()
            for line in lines:
                points.append(float(line.strip().split('\t')[0]))
        return points

    def add_mir_eval(self):
        try:
            import mir_eval as me

            class MirEval(QaMetric):
                """
                mir_eval metrics wrapper. More information in
                https://craffel.github.io/mir_eval/#module-mir_eval.onset
                """
                def score(self, reference, estimated):
                    scores = me.onset.evaluate(reference, estimated)
                    return scores

            self.set_metrics(MirEval())
        except ImportError:
            print('Not using mir_eval metrics because the package was not found')
            pass  # Module doesn't exist, deal with it.
        pass

    def plot(self, name, plots_dir='plots', force=False):
        plot_name = '{}/{}'.format(plots_dir, name)
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)

        if os.path.exists('{}.png'.format(plot_name)) and not force:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

        except ImportError:
            print('Not plotting because matplotlib was not found')
            return

        def r():
            return np.random.randint(0, 255)

        audio = self.data[name]
        time = np.linspace(0, len(audio) / float(self.fs), len(audio))
        plt.figure()
        plt.plot(time, audio)
        plt.title(name)
        if name in self.ground_true:
            plt.axvline(x=self.ground_true[name][0], label='ground truth',
                        color='green', alpha=0.7)
            for x in self.ground_true[name][1:]:
                plt.axvline(x=x, color='green', alpha=0.7)

        for key_sol, solution in self.solutions.items():
            if not key_sol[1] == name:
                continue

            color = ('#%02X%02X%02X' % (r(), r(), r()))
            plt.axvline(x=solution[0], label=key_sol[0], color=color, alpha=0.7)
            for x in solution[1:]:
                plt.axvline(x=x, color=color, alpha=0.7)

        plt.legend()

        plt.savefig(plot_name)

    def plot_all(self, *args, **kwargs):
        for name in self.data.iterkeys():
            self.plot(name, *args, **kwargs)

    def generate_stats(self, output_file='stats.log'):
        text = []

        wrappers = self.wrappers.keys()
        data = self.data.keys()
        metrics = self.metrics.keys()

        stats = {}
        for w in wrappers:
            for m in metrics:
                if m == 'MirEval':
                    scores = {'F-measure': [],
                              'Precision': [],
                              'Recall': []}
                    for d in data:
                        d_me = self.scores[(w, d, m)]
                        scores['F-measure'].append(d_me['F-measure'])
                        scores['Precision'].append(d_me['Precision'])
                        scores['Recall'].append(d_me['Recall'])
                    stats[(w, 'F-measure', 'mean')] = np.mean(scores['F-measure'])
                    stats[(w, 'F-measure', 'std')] = np.std(scores['F-measure'])
                    stats[(w, 'Precision', 'mean')] = np.mean(scores['Precision'])
                    stats[(w, 'Precision', 'std')] = np.std(scores['Precision'])
                    stats[(w, 'Recall', 'mean')] = np.mean(scores['Recall'])
                    stats[(w, 'Recall', 'std')] = np.std(scores['Recall'])

                else:
                    d_me = []
                    for d in data:
                        d_me.append(np.mean(self.scores[(w, d, m)]))
                        stats[(w, m, 'mean')] = np.mean(d_me)
                        stats[(w, m, 'std')] = np.std(d_me)

        zip_metrics = set(zip(*stats.keys())[1])
        for w in wrappers:
            for m in zip_metrics:
                text.append('{} with {} has a mean value of {:.3f} with std of {:.3f}\n'
                            .format(w, m, stats[(w, m, 'mean')], stats[(w, m, 'std')]))

            text.append('*' * 70 + '\n')
        with open(output_file, 'w') as log_file:
            log_file.write(''.join(text))
