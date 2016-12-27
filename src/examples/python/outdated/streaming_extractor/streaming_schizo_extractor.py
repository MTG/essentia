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

#! /usr/bin/env python

import sys, os
from os.path import join
import numpy

import essentia
import essentia.standard as standard
import essentia.streaming as streaming
from essentia import Pool, INFO

from metadata     import readMetadata, getAnalysisMetadata
import replaygain
import lowlevel
import midlevel
import highlevel
import panning
import segmentation

essentia_usage = "usage: \'essentia_extractor [options] config_file input_soundfile output_results\'"

# global defines:
analysisSampleRate = 44100.0

def parse_args():

    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=essentia_usage, version=essentia_version)

    parser.add_option("-v","--verbose",
      action="store_true", dest="verbose", default=False,
      help="verbose mode")

    parser.add_option("-s","--segmentation",
      action="store_true", dest="segmentation", default=False,
      help="do segmentation")

    parser.add_option("-p","--profile",
      action="store", type="string", dest="profile", default="music",
      help="computation mode: 'music', 'sfx' or 'broadcast'")

    parser.add_option("--start",
      action="store", dest="startTime", default="0.0",
      help="time in seconds from which the audio is computed")

    parser.add_option("--end",
      action="store", dest="endTime", default="1.0e6",
      help="time in seconds till which the audio is computed, 'end' means no time limit")

    parser.add_option("--svmpath",
      action="store", dest="svmpath", default=join('..', 'svm_models'),
      help="path to svm models")

    (options, args) = parser.parse_args()

    return options, args


def computeAggregation(pool, segments_namespace=''):
    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']

    exceptions={'lowlevel.mfcc' : ['mean', 'cov', 'icov']}
    for namespace in segments_namespace:
        exceptions[namespace+'.lowlevel.mfcc']=['mean', 'cov', 'icov']

    if segments_namespace: exceptions['segmentation.timestamps']=['copy']
    return standard.PoolAggregator(defaultStats=stats,
                                   exceptions=exceptions)(pool)


def addSVMDescriptors(pool, pathToSvmModels):
  #svmModels = [] # leave this empty if you don't have any SVM models
  svmModels = ['BAL', 'CUL', 'GDO', 'GRO', 'GTZ', 'PS', 'VI',
               'MAC', 'MAG', 'MEL', 'MHA', 'MPA', 'MRE', 'MSA']

  for model in svmModels:
      modelFilename = join(pathToSvmModels, model+'.model')
      svm = standard.SvmClassifier(model=modelFilename)(pool)
      pool.merge(svm)

def computeLowLevel(input_file, neqPool, eqPool, startTime, endTime, namespace=''):
    llspace = 'lowlevel.'
    rhythmspace = 'rhythm.'
    if namespace :
        llspace = namespace + '.lowlevel.'
        rhythmspace = namespace + '.rhythm.'

    rgain, sampleRate, downmix = getAnalysisMetadata(neqPool)
    loader = streaming.EasyLoader(filename = input_file,
                                  sampleRate = sampleRate,
                                  startTime = startTime,
                                  endTime = endTime,
                                  replayGain = rgain,
                                  downmix = downmix)

    eqloud = streaming.EqualLoudness()
    loader.audio >> eqloud.signal
    lowlevel.compute(eqloud.signal, loader.audio, neqPool, startTime, endTime, namespace)
    lowlevel.compute(eqloud.signal, eqloud.signal, eqPool, startTime, endTime, namespace)
    essentia.run(loader)

    # check if we processed enough audio for it to be useful, in particular did
    # we manage to get an estimation for the loudness (2 seconds required)
    if not neqPool.containsKey(llspace + "loudness") and\
       not eqPool.containsKey(llspace + "loudness"):
        INFO('ERROR: File is too short (< 2sec)... Aborting...')
        sys.exit(2)

    sampleRate = neqPool['metadata.audio_properties.analysis_sample_rate']

    numOnsets = len(neqPool[rhythmspace + 'onset_times'])
    onset_rate = numOnsets/float(loader.audio.totalProduced())*sampleRate
    neqPool.set(rhythmspace + 'onset_rate', onset_rate)

    numOnsets = len(eqPool[rhythmspace + 'onset_times'])
    onset_rate = numOnsets/float(loader.audio.totalProduced())*sampleRate
    eqPool.set(rhythmspace + 'onset_rate', onset_rate)

def computeMidLevel(input_file, neqPool, eqPool, startTime, endTime, namespace=''):
    rgain, sampleRate, downmix = getAnalysisMetadata(neqPool)
    loader = streaming.EasyLoader(filename = input_file,
                                  sampleRate = sampleRate,
                                  startTime = startTime,
                                  endTime = endTime,
                                  replayGain = rgain,
                                  downmix = downmix)

    eqloud = streaming.EqualLoudness()
    loader.audio >> eqloud.signal
    midlevel.compute(loader.audio, neqPool, startTime, endTime, namespace)
    midlevel.compute(eqloud.signal, eqPool, startTime, endTime, namespace)
    essentia.run(loader)


if __name__ == '__main__':

    opt, args = parse_args()

    if len(args) != 2: #3:
        print "Incorrect number of arguments\n", essentia_usage
        sys.exit(1)


    #profile = args[0]
    input_file = args[0]
    output_file = args[1]

    neqPool = Pool()
    eqPool = Pool()
    startTime = float(opt.startTime)
    endTime = float(opt.endTime)

    # compute descriptors

    readMetadata(input_file, eqPool)
    INFO('Process step 1: Replay Gain')
    replaygain.compute(input_file, eqPool, startTime, endTime)

    segments_namespace=[]
    neqPool.merge(eqPool, 'replace')
    if opt.segmentation:
        INFO('Process step 2: Low Level')
        computeLowLevel(input_file, neqPool, eqPool, startTime, endTime)
        segmentation.compute(input_file, eqPool, startTime, endTime)
        segments = eqPool['segmentation.timestamps']
        for i in xrange(len(segments)-1):
            startTime = segments[i]
            endTime = segments[i+1]

            INFO('**************************************************************************')
            INFO('Segment ' + str(i) + ': processing audio from ' + str(startTime) + 's to ' + str(endTime) + 's')
            INFO('**************************************************************************')

            # set segment name:
            segment_name = 'segment_'+ str(i)
            neqPool.set('segments.'+segment_name+'.name', segment_name)
            eqPool.set('segments.'+segment_name+'.name', segment_name)
            # set segment scope:
            neqPool.set('segments.'+segment_name+'.scope', numpy.array([startTime, endTime]))
            eqPool.set('segments.'+segment_name+'.scope', numpy.array([startTime, endTime]))
            # compute descriptors:
            namespace = 'segments.'+segment_name+'.descriptors'
            segments_namespace.append(namespace)
            INFO('\tProcess step 2: Low Level')
            computeLowLevel(input_file, neqPool, eqPool, startTime, endTime, namespace)
            INFO('\tProcess step 3: Mid Level')
            computeMidLevel(input_file, neqPool, eqPool, startTime, endTime, namespace)
            INFO('\tProcess step 4: High Level')
            highlevel.compute(eqPool, namespace)
            highlevel.compute(neqPool, namespace)

        # compute the rest of the descriptors for the entire audio. LowLevel
        # descriptors were already computed during segmentation
        startTime = float(opt.startTime)
        audio_length = neqPool['metadata.audio_properties.length']
        endTime = float(opt.endTime)
        if endTime > audio_length: endTime = audio_length
        INFO('**************************************************************************')
        INFO('processing entire audio from ' + str(startTime) + 's to ' + str(endTime) + 's')
        INFO('**************************************************************************')
        INFO('Process step 3: Mid Level')
        computeMidLevel(input_file, neqPool, eqPool, startTime, endTime)
        INFO('Process step 4: High Level')
        highlevel.compute(eqPool)
        highlevel.compute(neqPool)

    else:
        INFO('Process step 2: Low Level')
        computeLowLevel(input_file, neqPool, eqPool, startTime, endTime)
        INFO('Process step 3: Mid Level')
        computeMidLevel(input_file, neqPool, eqPool, startTime, endTime)
        INFO('Process step 4: High Level')
        highlevel.compute(eqPool)
        highlevel.compute(neqPool)

    # compute statistics
    INFO('Process step 5: Aggregation')
    neqStats = computeAggregation(neqPool, segments_namespace)
    eqStats = computeAggregation(eqPool, segments_namespace)

    # svm
    #addSVMDescriptors(neqStats, opt.svmpath)

    # output results to file
    eqStats.set('metadata.audio_properties.equal_loudness', 1)
    neqStats.set('metadata.audio_properties.equal_loudness', 0)
    INFO('writing results to ' + output_file)
    standard.YamlOutput(filename=output_file)(eqStats)
    (name, ext) = os.path.splitext(output_file)
    output_file = name+'..neq.sig'
    INFO('writing results to ' + output_file)
    standard.YamlOutput(filename=output_file)(neqStats)

