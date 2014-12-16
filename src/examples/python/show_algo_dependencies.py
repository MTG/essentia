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


import subprocess

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


    print "---------- %s : %s ----------" % (mode, algo)
    
    algos = sorted(list(set(algos) - set([(mode, algo)])))
    print "Dependencies:"
    for m,a in algos:
        print m + '\t' + a
    print


    print '\n'.join(lines)
    print 
    print


import essentia.standard as es
for algo in es.algorithmNames():
    find_dependencies('standard', algo)


import essentia.streaming as es
for algo in es.algorithmNames():
    find_dependencies('streaming', algo)

