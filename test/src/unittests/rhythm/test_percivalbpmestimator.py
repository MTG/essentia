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

# This "empty" file is included here as a "dummy" returning FAIL.
# The to open the Pull Request for Percival Unit Tests as draft to get started. 
# DELETE ABOVE TWO LINES WHEN FINISHED


from essentia_test import *

class TestPercivalBpmEstimator(TestCase):
    def testDummy(self):
        self.assertEqual(1,0)


suite = allTests(TestPercivalBpmEstimator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

