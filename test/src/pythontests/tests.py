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
