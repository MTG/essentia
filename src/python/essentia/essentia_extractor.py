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

import essentia
import logging
import sys
import os
import yaml
import numpy
from essentia.extractor import segmentation
from essentia import INFO


def mergeRecursiveDict(d1, d2):
    '''This function merges the values contained in d2 into d1.
    If a value was already in d1, it overwrites that value.'''
    for (key, value) in d2.items():
        if key in d1 and isinstance(value, dict):
            mergeRecursiveDict(d1[key], value)
        else:
            d1[key] = value
    return d1


def toList(obj):
    if obj is None: return []
    if isinstance(obj, list): return obj
    return [ obj ]


def getOptionsFor(opts, name):
    '''Takes a global options map and a name of a specific extractor, and
    returns a map containing only the options for the specified extractor.'''

    result = opts.copy()

    try: del result['specific']
    except KeyError: pass

    try: mergeRecursiveDict(result, opts['specific'][name])
    except KeyError: pass

    return result


def loadAudioFile(inputFilename, pool, options):

    sampleRate = options['sampleRate']

    audio = essentia.MonoLoader(filename = inputFilename,
                                sampleRate = sampleRate,
                                downmix = 'mix')()

    #pool.setCurrentNamespace('metadata')

    # compute the temporal duration
    duration = essentia.Duration(sampleRate = options['sampleRate'])(audio)

    # trim audio if asked
    startTime = options['startTime']
    endTime = options['endTime']
    if startTime >= endTime:
       raise essentia.EssentiaError('In the configuration file, startTime should be lower or equal than endTime')
    startSample = int(options['sampleRate'] * startTime)
    try:
       endSample = int(options['sampleRate'] * endTime)
    except TypeError:
       endTime = duration
       endSample = int(options['sampleRate'] * duration)

    if startTime > duration:
       raise essentia.EssentiaError('The file is too short to be trimmed from second %d to second %d' % (startTime, endTime))
    else:
       if endTime > duration:
          if startTime != 0.0:
             INFO('The file is being trimmed from second %d to second %d' % (startTime, duration))
             audio = audio[startSample:]
       else:
          if startTime != 0.0 or endTime != duration:
             INFO('The file is being trimmed from second %d to second %d' % (startTime, endTime))
             audio = audio[startSample:endSample]

    #pool.setGlobalScope([ 0.0, len(audio) / options['sampleRate'] ])
    pool.add('metadata.duration', duration)#, pool.GlobalScope)
    pool.add('metadata.duration_processed', len(audio) / options['sampleRate'])#, pool.GlobalScope)

    # add sample rate and number of channels to pool
    pool.add('metadata.filename', inputFilename)#, pool.GlobalScope)
    pool.add('metadata.sample_rate', sampleRate)##, pool.GlobalScope)
    #pool.add('channels', originalChannelsNumber, pool.GlobalScope)

    return audio

def descriptorNames(descList, namespace=''):
    if not namespace: return [desc.split('.')[1] for desc in descList]
    descNames = []
    for desc in descList:
        key = desc.split('.')[0]
        value = desc.split('.')[1]
        if key == namespace: descNames.append(value)
    return descNames


def preProcess(audio, pool, options, namespace=''):
    # which preprocessing preprocessing do we want to apply?
    preprocessing = toList(options['preprocessing'])

    # filtering and normalization
    for step in preprocessing:
        # do we remove the DC component?
        if step == 'dckiller':
            audio = essentia.DCRemoval()(audio)

        # do we normalize the audio?
        elif step == 'normalize':
            # compute replay gain first
            replayGain = essentia.ReplayGain(sampleRate = options['sampleRate'])(audio)
            pool.add(namespace + '.' + 'replay_gain', replayGain)#, pool.GlobalScope)

            # rescale audio if not silent (also apply a 6dB pre-amplification)
            if replayGain < 68.0:
                audio = essentia.Scale(factor = 10**((replayGain)/20))(audio)

        # do we apply an equal-loudness filter on all the audio?
        elif step == 'eqloud':
            audio = essentia.EqualLoudness(sampleRate = options['sampleRate'])(audio)

        else:
            raise essentia.EssentiaError('Unknown preprocessing step: \'%s\'' % step)

    return audio


def computeExtractor(name, audio, pool, options):

    if name in options['computed']:
        return

    try:
        exec('import extractor_local.' + name + ' as extractor')
    except:
        exec('import extractor.' + name + ' as extractor')

    # make sure all dependencies are satisfied
    for dep in toList(extractor.dependencies):
        computeExtractor(dep, audio, pool, options)

    # build specific options map
    opts = getOptionsFor(options, name)

    # sets specific verbosity level and current pool namespace
    logger = logging.getLogger()
    if opts['verbose']: logger.setLevel(logging.INFO)
    else:               logger.setLevel(logging.WARNING)

    #pool.setCurrentNamespace(extractor.namespace)

    # apply specific preprocessing
    try:
        preprocessedAudio = preProcess(audio, pool, opts, extractor.namespace)
    except KeyError:
        preprocessedAudio = audio

    # do the actual computation
    descriptorsBefore = descriptorNames(pool.descriptorNames(), extractor.namespace)
    extractor.compute(preprocessedAudio, pool, opts)
    descriptorsAfter = descriptorNames(pool.descriptorNames(), extractor.namespace)

    options['computed'].append(name)
    options['generatedBy'][name] = [ desc for desc in descriptorsAfter if desc not in descriptorsBefore ]


def computeAllExtractors(extractors, audio, pool, options):

    # process all extractors, one by one
    for name in extractors:
        try:
            computeExtractor(name, audio, pool, options)
        except Exception:
            print 'ERROR: when trying to compute', name, 'features'
            raise

def percentile(values, p):
    values.sort()
    idx = int(p/100.0 * len(values))
    return values[idx]



def cleanStats(pool, options):

    # remove unwanted descriptors
    wantedStats = {}
    supportedStats = ['mean', 'min', 'max', 'var', 'dmean', 'dvar', 'dmean2',\
                      'dvar2', 'value', 'copy', 'single_gaussian', 'cov', 'icov']

    for extractor in options['specific']:

        if 'output' in options['specific'][extractor] and extractor in options['generatedBy']:
            outputList = options['specific'][extractor]['output']
            exec('import extractor.' + extractor + ' as extractor_module')
            namespace = extractor_module.namespace

            # check if we're not asking for some inexistent descriptor
            for descriptor in outputList:
                generated = options['generatedBy'][extractor]
                if descriptor not in generated:
                    raise essentia.EssentiaError('Could not find descriptor \'' + descriptor + '\'. Available are: \'' + '\', \''.join(generated) + '\'')

            for descriptor in options['generatedBy'][extractor]:

                if descriptor not in outputList:
                    #del pool.descriptors[namespace][descriptor]
                    pool.remove(namespace + '.' + descriptor)
                else:
                    try:
                        wantedStats[namespace + '.' + descriptor] = options['specific'][extractor]['output'][descriptor]
                    except KeyError:
                        wantedStats[namespace] = {}
                        wantedStats[namespace + '.' + descriptor] = options['specific'][extractor]['output'][descriptor]
            for (k,v) in wantedStats.items():
                if not isinstance(v, list):
                    wantedStats[k] = [v]
                stats = wantedStats[k]
                unwantedStats = []
                for stat in stats:
                    if stat not in supportedStats:
                        unwantedStats += [stat]
                        print 'Ignoring', stat, 'for', k, '. It is not supported.'
                    if stat == 'single_gaussian':
                        unwantedStats += [stat]
                        wantedStats[k] += ['mean', 'cov', 'icov']
                for stat in unwantedStats: wantedStats[k].remove(stat)

    metaDescs = descriptorNames(pool.descriptorNames(), 'metadata')
    wantedStats['lowlevel.spectral_contrast.mean'] = ['copy']
    wantedStats['lowlevel.spectral_contrast.var'] = ['copy']
    for desc in metaDescs:
        wantedStats['metadata' + '.' + desc] = ['copy']

    return wantedStats


def computeSegments(audio, segments, extractors, megalopool, options):

    sampleRate = options['sampleRate']

    for segment in segments:

        segmentName = 'segment_' + str("%02d" % segments.index(segment))

        if options['verbose']:
            print 'Processing', segmentName, 'from second', segment[0], 'to second', segment[1]

        # creating pool...
        poolSegment = essentia.Pool()

        # creating audio segment
        audioSegment = audio[segment[0] * sampleRate : segment[1] * sampleRate]

        # creating the pool
        #poolSegment.setCurrentNamespace('metadata')
        #poolSegment.setGlobalScope([ 0.0, len(audioSegment) / sampleRate ])
        poolSegment.add('metadata.duration_processed', float(len(audioSegment)) / sampleRate)#, poolSegment.GlobalScope)

        # process all extractors
        options['computed'] = []
        computeAllExtractors(extractors, audioSegment, poolSegment, options)

        # remove unwanted descriptors
        wantedStats = cleanStats(poolSegment, options)

        # adding to megalopool
        segmentScope = [segment[0], segment[1]]
        poolSegmentAggregation = essentia.PoolAggregator(exceptions=wantedStats)(poolSegment)
        #megalopool.add(segmentName, poolSegment.aggregate_descriptors(wantedStats))#, segmentScope)
        addToPool(segmentName, megalopool, poolSegmentAggregation)

    return megalopool

def addToPool(name, a, b):
    # adds data in pool b into pool a under name name
    descriptors = b.descriptorNames()
    for descriptor in descriptors:
        a.add(name + '.' + descriptor, b.value(descriptor))

def spectral_contrast_stats(pool, sc_name, stats):
    if sc_name  in pool.descriptorNames():
        sc_stats = []
        if sc_name in stats: sc_stats = stats[sc_name]
        else: sc_stats = ['mean', 'var']
        sc_contrast = pool.value(sc_name)[0]
        pool.remove(sc_name)
        if 'mean' in sc_stats: pool.add(sc_name + '.mean', numpy.mean(sc_contrast, axis=0))
        if 'var'  in sc_stats: pool.add(sc_name + '.var',  numpy.var(sc_contrast, axis=0))
        if 'cov'  in sc_stats: pool.add(sc_name + '.cov',  numpy.cov(sc_contrast))

def compute(profile, inputFilename, outputFilename, userOptions = {}):

    # load profile
    profileDirectory = __file__.split(os.path.sep)[:-1]
    profileDirectory.append('profiles')
    profileDirectory.append('%s_config.yaml' % profile)

    try:
        # try to load the predefined profile, if it exists
        config = open(os.path.sep.join(profileDirectory), 'r').read()
    except:
        # otherwise, just load the file that was specified
        config = open(profile, 'r').read()

    options = yaml.load(config)
    mergeRecursiveDict(options, userOptions)

    # which format for the output?
    format = options['outputFormat']
    if format not in [ 'xml', 'yaml' ]:
        raise essentia.EssentiaError('output format should be either \'xml\' or \'yaml\'')
    if format == 'xml':
        xmlOutput = True
    else:
        xmlOutput = False

    # we need this for dependencies checking
    options['computed'] = []
    options['generatedBy'] = {}

    # get list of extractors to compute
    extractors = options['extractors']

    # create pool & megalopool
    pool = essentia.Pool()

    # load audio file into memory
    audio = loadAudioFile(inputFilename, pool, options)

    # preprocess audio by applying a DC filter, normalization, etc...
    # preprocessing is a special step because it modifies the audio, hence it
    # must be executed before all the other extractors
    audio = preProcess(audio, pool, options, 'metadata')
    options['globalPreprocessing'] = options['preprocessing']
    del options['preprocessing']

    # process all extractors
    computeAllExtractors(extractors, audio, pool, options)

    # process segmentation if asked
    if options['segmentation']['doSegmentation']:
        segments = segmentation.compute(inputFilename, audio, pool, options)

    # remove unwanted descriptors
    wantedStats = cleanStats(pool, options)

    # add to megalopool
    #megalopool = essentia.Pool()
    scope = [ 0.0, len(audio)/options['sampleRate'] ]
    #megalopool.add('global', pool.aggregate_descriptors(wantedStats))#, scope)
    megalopool = essentia.PoolAggregator(exceptions=wantedStats)(pool)
    # special case for spectral contrast, which is only 1 matrix, therefore no
    # stats are computed:
    spectral_contrast_stats(megalopool, 'lowlevel.spectral_contrast', wantedStats)

    # plotting descriptors evolution
    try:
        if options['plots']:
            import plotting
            plotting.compute(inputFilename, audio, pool, options)
    except KeyError: pass

    # compute extractors on segments
    if options['segmentation']['doSegmentation']:
        if options['segmentation']['computeSegments']:
            if len(segments) == 0:
                megalopool.add('void', [0])
            else:
                computeSegments(audio, segments, extractors, megalopool, options)

    # save to output file
    essentia.YamlOutput(filename=outputFilename)(megalopool)
    #megalopool.save(outputFilename, xml = xmlOutput)
