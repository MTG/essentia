#! /bin/sh

set -x
set -e

function scons_cleaner () {
find . \( -name '*\.pyc' -o -name .sconsign.dblite \) -exec rm -rf {} \;
rm -f src/algorithms/_registration/essentia_algorithms_reg.cpp
rm -f src/python/essentia/__svn_version__.py
}

## clean
scons -c check_filenames $*
scons -c check_test_filenames $*
scons -c doxygen $*
scons -c doc $*
scons -c test $*
scons -c python $*
scons -c $*
scons_cleaner

## build
scons $*
scons python $*
scons test $*

## run tests
scons run_tests $* #OnlySomeTests
scons run_python_tests $*

## python install
scons python_install $*

## doc
scons doc $*

## check
scons check_filenames $*
scons check_test_filenames $*
