#!/usr/bin/env python

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



from essentia_test import *
from viterbi_trellis import ViterbiTrellis

class TestViterbi(TestCase):


    def testEmpty(self):
        p1  = [[],[]]
        p2  = []
        p3 = [] 
        p4 = []
        p5 = []
        emptyOutput = Viterbi()(p1,p2,p3,p4,p5)
        print(emptyOutput)


    # Inpute parameters are
    """
        observationProbabilities (vector_vector_real) - the observation probabilities
        initialization (vector_real) - the initialization
        fromIndex (undefined) - the transition matrix from index
        toIndex (undefined) - the transition matrix to index
        transitionProbabilities (vector_real) - the transition probabilities matrix
    """

    def testRegression(self):
        # Example from https://pypi.org/project/viterbi-trellis/ 
        p1  = [[2, 6, 4], [4, 6], [0, 2, 6]]
        v = ViterbiTrellis(p1,  lambda x: x / 2.0, lambda x, y: abs(y - x))
        best_path = v.viterbi_best_path() # result is [2, 0, 1]

        p4 = []
        p5 = []
        output = Viterbi()(p1,lambda x: x / 2.0, lambda x, y: abs(y - x),p4,p5)
        print(output)



suite = allTests(TestViterbi)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
