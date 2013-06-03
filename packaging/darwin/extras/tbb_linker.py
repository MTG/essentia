#!/usr/bin/env python

# this script is used in the postinstall stage to create dynamic linking for
# tbb src and libs

import os

essentia_third_party = '/essentia/third_party/'
tbb_mac = essentia_third_party + 'tbb20_020oss_mac/ia32/cc4.0.1_os10.4.9/'
tbb_src = essentia_third_party + 'tbb20_020oss_src/'

def link(source, target):
    cmd = 'sudo ln -s' + ' ' + source + ' ' + target
    print cmd
    return os.system(cmd);

def link_tbb_files() :
    source = [tbb_src + 'include/tbb',
              tbb_mac + 'lib/libtbb.dylib',
              tbb_mac + 'lib/libtbb_debug.dylib',
              tbb_mac + 'lib/libtbbmalloc.dylib',
              tbb_mac + 'lib/libtbbmalloc_debug.dylib']

    target = [essentia_third_party + 'include/tbb',
              essentia_third_party + 'lib/libtbb.dylib',
              essentia_third_party + 'lib/libtbb_debug.dylib',
              essentia_third_party + 'lib/libtbbmalloc.dylib',
              essentia_third_party + 'lib/libtbbmalloc_debug.dylib']

    ret = 0
    for i in range(len(source)):
        ret += link(source[i], target[i])

    return ret


if __name__ == '__main__':
    ret = link_tbb_files();
    print "returning ", ret
    exit(ret)
