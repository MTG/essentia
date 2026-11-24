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

class TestFrameToReal(TestCase):

   def testInvalidParam(self):
        self.assertConfigureFails(FrameToReal(), {'frameSize': -1})
        self.assertConfigureFails(FrameToReal(), {'hopSize': -1})
        self.assertConfigureFails(FrameToReal(), {'frameSize': 0})
        self.assertConfigureFails(FrameToReal(), {'hopSize': 0})
   
   # extreme conditions
   def testEmpty(self):
        self.assertRaises(RuntimeError, lambda: FrameToReal()([]))

   def testHopSize(self):
        found = FrameToReal(frameSize=100, hopSize = 60)(ones(1024))
        self.assertEqual(len(found), 60)


   def testSmallHopSize(self):
        found = FrameToReal(frameSize=10000, hopSize = 1)(zeros(1024))
        self.assertEqual(len(found), 1)

suite = allTests(TestFrameToReal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
