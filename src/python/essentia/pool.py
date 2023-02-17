# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

from itertools import izip
import numpy
import essentia
from essentia import EssentiaError
import numpy.core.arrayprint

# make it so we don't see summarized output, we want the full stuff
numpy.core.arrayprint.set_printoptions(precision=10,
                                       linewidth=80,
                                       threshold=1e6,
                                       suppress=True)


def percentile(values, p):
    values.sort()
    idx = int(p/100.0 * len(values))
    return values[idx]


# a simple pool of numbers with save to audioclas format and save to textfile
class Pool:
    GlobalScope = -1   # this value doesn't really matter, we just need the identifier
    CurrentScope = -2  # this value doesn't really matter, we just need the identifier

    def __init__(self):
        self.descriptors = {}
        self.statsBlackList = []

        self.__currentNamespace = 'AnonymousNamespace'
        self.__globalScope = [0.0, 0.0]
        self.__currentScope = [0.0, 0.0]

    def setCurrentNamespace(self, namespace):
        self.__currentNamespace = namespace
        if namespace not in self.descriptors:
            self.descriptors[namespace] = {}

    def ns(self):
        return self.__currentNamespace

    def setGlobalScope(self, scope):
        self.__globalScope = scope

    def setCurrentScope(self, scope):
        self.__currentScope = scope

    def add(self, name, value, scope=CurrentScope):
        if isinstance(value, tuple):
            raise EssentiaError('error when adding %s: you can\'t add tuples to the pool' % name)

        if scope == self.GlobalScope:
            self.statsBlackList.append((self.__currentNamespace, name))
            scope = self.__globalScope

        elif scope == self.CurrentScope:
            scope = self.__currentScope

        try:
            self.descriptors[self.__currentNamespace][name]['values'].append(value)
            self.descriptors[self.__currentNamespace][name]['scopes'].append(scope)
        except KeyError:
            self.descriptors[self.__currentNamespace][name] = {'values': [value], 'scopes': [scope]}

    def mean_scope(self, scopeFrom, scopeTo):
        descriptors_mean = {}

        for key in self.descriptors.keys():
            descriptor = self.descriptors[self.__currentNamespace][key]
            values_in_scope = []

            # Global descriptor (should also check that scope spans the entire file)
            if len(descriptor['values']) == 1:
                descriptors_mean[key] = descriptor['values'][0]
                continue

            for scope, value in zip(descriptor['scopes'], descriptor['values']):
                if scope[0] >= scopeFrom and scope[1] <= scopeTo:
                    values_in_scope.append(value)

            if len(values_in_scope) > 0:
                try:
                    descriptors_mean[key] = essentia.array(numpy.mean(values_in_scope, axis=0))
                except TypeError:  # values are not numeric
                    descriptors_mean[key] = values_in_scope[0]

        return descriptors_mean

    def var_scope(self, scopeFrom, scopeTo):
        descriptors_var = {}

        for key in self.descriptors.keys():
            descriptor = self.descriptors[self.__currentNamespace][key]
            values_in_scope = []

            # Global descriptor (should also check that scope spans the entire file)
            if len(descriptor['values']) == 1:
                descriptors_var[key] = 0.0
                continue

            for scope, value in zip(descriptor['scopes'], descriptor['values']):
                if scope[0] >= scopeFrom and scope[1] <= scopeTo:
                    values_in_scope.append(value)

            if len(values_in_scope) > 0:
                try:
                    descriptors_var[key] = essentia.array(numpy.var(values_in_scope, axis=0))
                except TypeError:  # values are not numeric
                    descriptors_var[key] = 0.0

        return descriptors_var

    def aggregate_descriptors(self, descriptors_stats={}):
        aggregated = {}
        for namespace in self.descriptors:
            aggregated[namespace] = {}
            descs = self.descriptors[namespace].keys()
            descs.sort()

            stats_default = ['mean', 'var', 'min', 'max']

            for desc in descs:
                values = self.descriptors[namespace][desc]['values']

                if (namespace, desc) in self.statsBlackList:
                    # make sure there is only one value
                    if len(values) != 1:
                        raise EssentiaError('You declared %s as a global descriptors, but there are more than 1 value: %s' % (desc, values))

                    value = values[0]
                    try:
                        # if value is numeric
                        aggregated[namespace][desc] = {'value': essentia.array(value)}
                    except:
                        # if value is not numeric
                        aggregated[namespace][desc] = {'value': value}

                    continue

                aggregated[namespace][desc] = {}
                aggrDesc = aggregated[namespace][desc]

                stats = list(stats_default)
                if not isinstance(values[0], numpy.ndarray):
                    #stats += [ 'percentile_5', 'percentile_95' ]
                    stats += ['dmean', 'dmean2', 'dvar', 'dvar2']

                if namespace in descriptors_stats and desc in descriptors_stats[namespace]:
                    stats = descriptors_stats[namespace][desc]

                try:

                    if 'mean' in stats:
                        aggrDesc['mean'] = essentia.array(numpy.mean(values, axis=0))

                    if 'var' in stats:
                        aggrDesc['var'] = essentia.array(numpy.var(values, axis=0))

                    if 'min' in stats:
                        aggrDesc['min'] = essentia.array(numpy.min(values, axis=0))

                    if 'max' in stats:
                        aggrDesc['max'] = essentia.array(numpy.max(values, axis=0))

                    derived = None
                    derived2 = None

                    if 'dmean' in stats:
                        if not derived:
                            derived = [a - b for a, b in izip(values[1:], values[:-1])]
                        aggrDesc['dmean'] = essentia.array(numpy.mean(numpy.abs(derived), axis=0))

                    if 'dvar' in stats:
                        if not derived:
                            derived = [a - b for a, b in izip(values[1:], values[:-1])]
                        aggrDesc['dvar'] = essentia.array(numpy.var(derived, axis=0))

                    if 'dmean2' in stats:
                        if not derived:
                            derived = [a - b for a, b in izip(values[1:], values[:-1])]
                        if not derived2:
                            derived2 = [a - b for a, b in izip(derived[1:], derived[:-1])]
                        if derived2:
                            aggrDesc['dmean2'] = essentia.array(numpy.mean(numpy.abs(derived2), axis=0))
                        else:
                            aggrDesc['dmean2'] = 'undefined'

                    if 'dvar2' in stats:
                        if not derived:
                            derived = [a - b for a, b in izip(values[1:], values[:-1])]
                        if not derived2:
                            derived2 = [a - b for a, b in izip(derived[1:], derived[:-1])]
                        if derived2:
                            aggrDesc['dvar2'] = essentia.array(numpy.var(derived2, axis=0))
                        else:
                            aggrDesc['dvar2'] = 'undefined'


                    if 'frames' in stats:
                        aggrDesc['frames'] = essentia.array(values)

                    if 'single_gaussian' in stats:
                        single_gaussian = essentia.SingleGaussian()
                        (m, cov, icov) = single_gaussian(essentia.array(values))
                        aggrDesc['mean'] = m
                        aggrDesc['cov'] = cov
                        aggrDesc['icov'] = icov

                    for stat in stats:
                        if stat.startswith('percentile_'):
                            p = float(stat.split('_')[1])
                            aggrDesc[stat] = essentia.array(percentile(values, p))


                except (TypeError, ValueError):  # values are not numeric

                    if len(values) == 1:
                       aggrDesc['value'] = values[0]
                    else:
                       aggrDesc['value'] = []
                       for value in values:
                           aggrDesc['value'].append(value)

        return aggregated
