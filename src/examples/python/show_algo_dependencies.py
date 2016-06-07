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

import sys
import subprocess
import essentia.standard
import essentia.streaming
import yaml


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

    # function to assign nested dict elements by a list of nested keys
    def set_val(d, keys, val):
        reduce(lambda x,y: x[y], keys[:-1], d)[keys[-1]] = val

    algos = []
    lines = []
    tree = {}
    previous_key = []
    previous_indent = -8

    # NOTE: the code relies heavily on indentification of output in Essentia's logger
    for line in stdout:
        if line.startswith("[Factory   ] "):
        
            line = line.replace("[Factory   ] ", "")
            if line.count("Streaming: Creating algorithm: "):
                tab, a = line.split("Streaming: Creating algorithm: ")
                m = "streaming"
            elif line.count("Standard : Creating algorithm: "):
                tab, a = line.split("Standard : Creating algorithm: ")
                m = "standard"
            else: 
                continue

            lines.append(line)
            algos.append((m, a))
            
            indent = len(tab)

            if indent < previous_indent:
                previous_key = previous_key[:-2]
            if indent == previous_indent:
                previous_key = previous_key[:-1]

            set_val(tree, previous_key + [(m,a)], {})
            previous_key += [(m, a)]          

            previous_indent = indent
 
    algos = sorted(list(set(algos) - set([(mode, algo)])))
    return algos, tree, lines


def print_dependencies(algos, tree=None, lines=None):
    print "Dependencies:"
    for m,a in algos:
        print m + '\t' + a
    print

    if tree:
        print "Dependencies tree (yaml)"
        print yaml.safe_dump(tree, indent=4)


    if lines:
        print "Essentia logger output"
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
    
    dependencies, tree, _ = find_dependencies(mode, algo)  
    print_dependencies(dependencies, tree)

else:
    # search dependencies non-recursively for all algorithms
    for mode in ['standard', 'streaming']:
        for algo in algos[mode]:
            print "---------- %s : %s ----------" % (mode, algo)
            print_dependencies(*find_dependencies(mode, algo))
