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
