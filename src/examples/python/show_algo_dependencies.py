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

import argparse



def find_dependencies(mode, algo):
    if algo in ['MusicExtractor', 'FreesoundExtractor']:
        # FIXME These are special algorithms that instantiate all dependencies
        # inside compute(). Ideally, they should be rewritten to do that in
        # configure() methods. Feeding a test audiofile as an input.
        print("Running compute() for", algo)

        code = """
import essentia.%s as es
import essentia
from essentia.pytools.io import test_audiofile
test_audio = test_audiofile()

essentia.log.infoActive = True
essentia.log.debugLevels += essentia.EFactory
es.%s()(test_audio)
""" % (mode, algo)
    else:
        code = """
import essentia.%s as es
import essentia

essentia.log.infoActive = True
essentia.log.debugLevels += essentia.EFactory
loader = es.%s()
""" % (mode, algo)

    proc = subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE)
    stderr = proc.communicate()[1].decode('utf8').split('\n')

    # function to assign nested dict elements by a list of nested keys
    def set_val(d, keys, val):
        from functools import reduce # Python 3 compatibility
        reduce(lambda x,y: x[y], keys[:-1], d)[keys[-1]] = val

    algos = []
    lines = []
    tree = {}
    previous_key = []
    previous_indent = -8

    # NOTE: the code relies heavily on indentification of output in Essentia's logger
    for line in stderr:
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
 
    algos = sorted(list(set(algos)))
    #algos = sorted(list(set(algos) - set([(mode, algo)])))
    return algos, tree, lines


def print_dependencies(algos, tree=None, lines=None):
    print("Dependencies:")
    for m,a in set(algos):
        print(m + '\t' + a)
    print('')

    if tree:
        print("Dependencies tree (yaml)")
        print(yaml.safe_dump(tree, indent=4))


    if lines:
        print("Essentia logger output")
        print('\n'.join(lines))
        print('')
        print('')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze Essentia's algorithm dependencies.")

    parser.add_argument("-a", "--algorithm", dest="algo", 
                                             help="algorithm to inspect",
                                             action="append",
                                             choices=set(essentia.standard.algorithmNames() + essentia.streaming.algorithmNames()))
    parser.add_argument("-m", "--mode", dest="mode", 
                                        help="mode (streaming, standard)", 
                                        choices=set(("standard", "streaming")))

    args = vars(parser.parse_args())

    streaming = essentia.streaming.algorithmNames()
    standard = essentia.standard.algorithmNames()

    print("Found %d streaming algorithms" % len(streaming))
    print("Found %d standard algorithms" % len(standard))
    print("%d algorithms in with both modes" % len(set(streaming) & set(standard)))
    print("%d algorithms in total" % len(set(streaming) | set(standard)))
    print('')

    algos = [(a, "standard") for a in standard] + [(a, "streaming") for a in streaming]

    if args['algo']:
        algos = [(a, m) for a, m in algos if a in args['algo']]
    else:
        print("Algorithm was not specified. Analyze dependencies for all algorithms")

    if args['mode']: 
        algos = [(a, m) for a, m in algos if m==args['mode']]
    else:
        print("Mode was not specified. Analyze dependencies for both modes")

    if not algos and args['algo'] and args['mode']:
        print('Algorithm "' + args['algo'] + '" not found in essentia.' + args['mode'])
        sys.exit()

    all_dependencies = []

    for algo, mode in algos:
        print("---------- %s : %s ----------" % (mode, algo))
        dependencies, tree, _ = find_dependencies(mode, algo)  
        #print_dependencies(dependencies, tree)
        print_dependencies(dependencies)
        all_dependencies += dependencies

    algos = sorted(list(set([a for m,a in all_dependencies])))
    print("The following %d algorithms will be required for building Essentia:" % len(algos))

    for a in algos:
        print(a)
