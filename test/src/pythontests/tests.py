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

#!/usr/bin/python

import os, sys

sys.path.append('../build/python/')
sys.path.append('../src/python/')

tests = ['test_essentia_import',
         'test_essentia_music',
        ]

passed = 0

for test in tests:
  print '+', test
  module = __import__( test, [], [], '*')
  return_value = module.test()
  if return_value == 0:
     passed += 1
     print '.'

print "Summary:"
print "        \033[1mExecuted Tests:         ", len(tests)
print "        \033[32;1mPassed Tests:           ", passed,
print "\033[0m"

if passed != len(tests): sys.exit(1)
