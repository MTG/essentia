#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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



from essentia_test import *
import os, os.path
import yaml


def loadTestData(type):
    transfo = GaiaTransform(history = 'highlevel/svm/test_svm_%s.history' % type)
    gt = yaml.load(open('highlevel/svm/test_svm_%s.gt.yaml' % type).read())
    return (transfo, gt)


def listFiles():
    allFiles = []
    for root, dirs, files in os.walk('highlevel/svm'):
        for f in files:
            allFiles.append(os.path.join(root, f))

    return [ f for f in allFiles if f.endswith('.sig') ]



try:
    g = GaiaTransform
    has_gaia = True
except NameError:
    has_gaia = False


class TestGaiaTransform(TestCase):

    def dotest(self, type):
        if not has_gaia:
            return

        testfiles = listFiles()
        transfo, gt = loadTestData(type)

        for f in testfiles:
            predicted = transfo(YamlInput(filename = f)())['genre']
            self.assertEquals(predicted, gt[os.path.split(f)[-1][:-4]])

    def testSingleDesc(self):
        self.dotest('singledesc')

    def testMultiDimDesc(self):
        self.dotest('multidimdesc')

    def testAll(self):
        self.dotest('all')

    def testProbability(self):
        if not has_gaia:
            return

        testfiles = listFiles()
        transfo, gt = loadTestData('probability')

        for f in testfiles:
            predicted = transfo(YamlInput(filename = f)())
            self.assertEquals(predicted['genre.value'], gt[os.path.split(f)[-1][:-4]])
            classes = [ d for d in predicted.descriptorNames() if d.startswith('genre.all') ]
            probs = [ predicted[d] for d in classes ]
            self.assertEquals(predicted['genre.probability'], max(probs))

    def testEmpty(self):
        pass

    def testInvalidParam(self):
        pass



suite = allTests(TestGaiaTransform)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
