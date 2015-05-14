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

import sys
import subprocess
import essentia.standard
import essentia.streaming


def find_dependencies(mode, algo):

    code = """
import essentia.%s as es
import essentia

essentia.log.infoActive = True
essentia.log.debugLevels += essentia.EFactory
loader = es.%s()
""" % (mode, algo)

    proc = subprocess.Popen(["python", "-c", code], stdout=subprocess.PIPE)
    stdout = proc.communicate()[0].split('\n')

    algos = []
    lines = []
    for line in stdout:

        if line.startswith("[Factory   ] "):
        
            if line.count("Streaming: Creating algorithm: "):
                a = line.split("Streaming: Creating algorithm: ")[-1]
                m = "streaming"
                lines.append(line)
                algos.append((m, a))

            if line.count("Standard : Creating algorithm: "):
                a = line.split("Standard : Creating algorithm: ")[-1]
                m = "standard"
                lines.append(line)
                algos.append((m, a))
    
    algos = sorted(list(set(algos) - set([(mode, algo)])))
    return algos, lines


def find_dependencies_tree(mode, algo):
    dependencies = find_dependencies(mode, algo)[0]
    results = {}
    for d in dependencies:
        results[d] = find_dependencies_tree(*d) 
    return results


def tree_to_list(dependencies):
    results = []
    for d in dependencies.keys():
        results += [d] + tree_to_list(dependencies[d])
    return results


def print_dependencies(algos, lines=None):
    print "Dependencies:"
    for m,a in algos:
        print m + '\t' + a
    print
    if lines:
        print '\n'.join(lines)
        print 
        print


try:
    algo = sys.argv[1]
    mode = sys.argv[2]
except:
    if len(sys.argv) > 1:
        print 'usage:', sys.argv[0], '[<algo_name> <streaming|standard>]'
        sys.exit()
    algo = None
    mode = None

algos = { 'standard': essentia.standard.algorithmNames(), 
          'streaming': essentia.streaming.algorithmNames() }


if algo: 
    # search dependencies recursively for algo
    try:
        if algo not in algos[mode]:
            print 'Algorithm "' + algo + '" not found in essentia.' + mode
            raise
    except:
        # mode != standard|streaming
        print 'usage:', sys.argv[0], '[<algo_name> <streaming|standard>]'
        sys.exit()

    print "---------- %s : %s ----------" % (mode, algo)
    
    dependencies = find_dependencies_tree(mode, algo)  
    print_dependencies(list(set(tree_to_list(dependencies))))

else:
    # search dependencies non-recursively for all algorithms
    for mode in ['standard', 'streaming']:
        for algo in algos[mode]:
            print "---------- %s : %s ----------" % (mode, algo)
            print_dependencies(*find_dependencies(mode, algo))
