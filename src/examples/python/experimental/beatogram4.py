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



import os, sys
from os.path import join

import essentia
from essentia.streaming import *
import essentia.standard as std

from pylab import median, mean, argmax
import matplotlib
#matplotlib.use('Agg') # in order to not grab focus on screen while batch processing
import matplotlib.pyplot as pyplot
import numpy as np
from numpy import shape, zeros, fabs

import scipy

# for key input
import termios, sys, os, subprocess
TERMIOS = termios

import copy

# for alsa
if sys.platform =='linux2':
    import wave, alsaaudio

import time
import thread

barkBands = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0,
              920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
              3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0,
              9500.0, 12000.0, 15500.0, 20500.0, 27000.0]

scheirerBands = [ 0.0, 50.0, 100.0, 150.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 5000.0, 10000.0]
EqBands = [20.0, 150.0, 400.0, 3200.0, 7000.0, 22000.0]
EqBands2 =[0.0, 75.0, 150.0, 400.0, 3200.0, 7000.0]


DOWNMIX ='mix'
# defines for novelty curve:
FRAMESIZE = 1024
HOPSIZE   = FRAMESIZE/2
WEIGHT='flat' #'supplied' #'flat'
SAMPLERATE=44100.0
WINDOW='hann' #'blackmanharris92'
BEATWINDOW=16 # number of beats where to compute statistics

# tempogram defines:
FRAMERATE = float(SAMPLERATE)/HOPSIZE
TEMPO_FRAMESIZE = 4;
TEMPO_OVERLAP=2;
STARTTIME = 0
ENDTIME = 2000

def computeOnsets(filename, pool):
    loader = EasyLoader(filename=filename,
                        sampleRate=pool['samplerate'],
                        startTime=STARTTIME, endTime=ENDTIME,
                        downmix=pool['downmix'])
    onset = OnsetRate()
    loader.audio >> onset.signal
    onset.onsetTimes >> (pool, 'ticks')
    onset.onsetRate >> None
    essentia.run(loader)
    pool.set('size', loader.audio.totalProduced())
    pool.set('length', pool['size']/pool['samplerate'])

def computeSegmentation(filename, pool):
    sampleRate = 44100
    frameSize = 2048
    hopSize = frameSize/2

    audio = EqloudLoader(filename = filename,
                       downmix=pool['downmix'],
                       sampleRate=sampleRate)

    fc = FrameCutter(frameSize=frameSize, hopSize=hopSize, silentFrames='keep')
    w = Windowing(type='blackmanharris62')
    spec = Spectrum()
    mfcc = MFCC(highFrequencyBound=8000)
    tmpPool = essentia.Pool()

    audio.audio >> fc.signal
    fc.frame >> w.frame >> spec.frame
    spec.spectrum >> mfcc.spectrum
    mfcc.bands >> (tmpPool, 'mfcc_bands')
    mfcc.mfcc>> (tmpPool, 'mfcc_coeff')

    essentia.run(audio)

    # compute transpose of features array, don't call numpy.matrix.transpose
    # because essentia fucks it up!!
    features = copy.deepcopy(tmpPool['mfcc_coeff'].transpose())
    segments = std.SBic(cpw=1.5, size1=1000, inc1=300, size2=600, inc2=50)(features)
    for segment in segments:
        pool.add('segments', segment*hopSize/sampleRate)
    #print pool['segments']

def computeNoveltyCurve(filename, pool):
    loader = EasyLoader(filename=filename,
                        sampleRate=pool['samplerate'],
                        startTime=STARTTIME, endTime=ENDTIME,
                        downmix=pool['downmix'])
    fc     = FrameCutter(frameSize=int(pool['framesize']),
                         silentFrames ='noise',
                         hopSize=int(pool['hopsize']),
                         startFromZero=False)
    window = Windowing(type=pool['window'],
                       zeroPhase=False)
    #freqBands = FrequencyBands(frequencyBands=EqBands, sampleRate=pool['samplerate'])
    freqBands = FrequencyBands(sampleRate=pool['samplerate'])
    spec = Spectrum()
    hfc = HFC()

    loader.audio >> fc.signal
    fc.frame >> window.frame >> spec.frame
    spec.spectrum >> freqBands.spectrum
    spec.spectrum >> hfc.spectrum
    freqBands.bands >> (pool, 'frequency_bands')
    hfc.hfc >> (pool, 'hfc')
    essentia.run(loader)

    pool.set('size', loader.audio.totalProduced())
    pool.set('length', pool['size']/pool['samplerate'])

    # compute a weighting curve that is according to frequency bands:
    frequencyBands = pool['frequency_bands']
    nFrames = len(frequencyBands)
    weightCurve= np.sum(frequencyBands, axis=0)
    weightCurve = [val/float(nFrames) for val in weightCurve]

    weightCurve = essentia.normalize(weightCurve)
    #pyplot.plot(weightCurve)
    #pyplot.show()


    noveltyCurve = std.NoveltyCurve(frameRate=pool['framerate'],
                                    weightCurveType=pool['weight'],
                                    weightCurve=weightCurve,
                                    normalize=False)(frequencyBands)
    #for x in noveltyCurve: pool.add('novelty_curve', x)
    #return

    # derivative of hfc seems to help in finding more precise beats...
    hfc = std.MovingAverage(size=int(0.1*pool['framerate']))(pool['hfc'])

    hfc = normalize(hfc)
    noveltyCurve = normalize(noveltyCurve)
    #noveltyCurve = essentia.normalize(noveltyCurve)
    dhfc = derivative(hfc)
    print max(hfc), max(noveltyCurve)
    for i, val in enumerate(dhfc):
        if val< 0: continue
        noveltyCurve[i] += 0.1*val

    # low pass filter novelty curve:
    env = std.Envelope(attackTime=0.001*pool['framerate'],
                       releaseTime=0.001*pool['framerate'])(noveltyCurve)

    # apply median filter:
    windowSize = 60./560.*pool['framerate'] #samples
    size = len(env)
    filtered = zeros(size, dtype='f4')
    for i in range(size):
        start = i-windowSize
        if start < 0: start = 0
        end = start + windowSize
        if end > size:
            end = size
            start = size-windowSize
        window = env[start:end]
        filtered[i] = env[i] - np.median(window) #max(np.median(window), np.mean(window))
        if filtered[i] < 0: filtered[i] = 0

    #pyplot.subplot(311)
    #pyplot.plot(noveltyCurve)
    #pyplot.subplot(312)
    #pyplot.plot(env, 'r')
    #pyplot.subplot(313)
    #pyplot.plot(filtered, 'g')
    #pyplot.show()

    #for x in noveltyCurve: pool.add('novelty_curve', x)
    #for x in filtered: pool.add('novelty_curve', x)
    #filtered = normalize(filtered)
    pool.set('novelty_curve', filtered)
    pool.set('original_novelty_curve', noveltyCurve)

def normalize(array):
    maxVal = max(array)
    if maxVal == 0: return zeros(len(array))
    return array/maxVal

def derivative(array):
    return scipy.diff(array)



def computeBeats(filename, pool):
    computeNoveltyCurve(filename, pool)
    recompute = True
    novelty = pool['novelty_curve']
    count = 0
    first_round = True
    bpmTolerance = 5
    minBpm = 30
    maxBpm =560
    while recompute:
        gen     = VectorInput(novelty)
        bpmHist = BpmHistogram(frameRate=pool['framerate'],
                               frameSize=pool['tempo_framesize'],
                               overlap=int(pool['tempo_overlap']),
                               maxPeaks=10,
                               windowType='hann',
                               minBpm=minBpm,
                               maxBpm=maxBpm,
                               normalize=False,
                               constantTempo=False,
                               tempoChange=5,
                               weightByMagnitude=True)

        gen.data >> bpmHist.novelty
        bpmHist.bpm            >> (pool, 'peaksBpm')
        bpmHist.bpmMagnitude   >> (pool, 'peaksMagnitude')
        bpmHist.harmonicBpm    >> (pool, 'harmonicBpm')
        bpmHist.harmonicBpm    >> (pool, 'harmonicBpm')
        bpmHist.confidence     >> (pool, 'confidence')
        bpmHist.ticks          >> (pool, 'ticks')
        bpmHist.ticksMagnitude >> (pool, 'ticksMagnitude')
        bpmHist.sinusoid       >> (pool, 'sinusoid')
        essentia.run(gen)

        print pool['peaksBpm']
        bpm = pool['harmonicBpm'][0]
        # align ticks with novelty curve
        #ticks, _ = alignTicks(pool['sinusoid'], pool['original_novelty_curve'], #novelty,
        #                      pool['framerate'], bpm, pool['length'])
        # or don't align ticks?
        ticks = pool['ticks']
        _, _, bestBpm= getMostStableTickLength(ticks)
        print 'estimated bpm:', bpm, 'bestBpm:', bestBpm, 'diff:', fabs(bpm-bestBpm)
        if first_round:
            pool.set('first_estimated_bpms', pool['peaksBpm'])
            first_round = False
        recompute = False
        if fabs(bestBpm - bpm) < bpmTolerance: recompute = False
        else:
            count+=1
            if count >= 5:
                bpmTolerance += 1
                count = 0
            print "recomputing!!!!"
            novelty = copy.deepcopy(pool['sinusoid'])
            pool.remove('sinusoid')
            pool.remove('novelty_curve')
            pool.remove('peaksBpm')
            pool.remove('peaksMagnitude')
            pool.remove('harmonicBpm')
            pool.remove('harmonicBpm')
            pool.remove('confidence')
            pool.remove('ticks')
            pool.remove('ticksMagnitude')
    #ticks = essentia.postProcessTicks(ticks)
    #ticks, ticksAmp = alignTicks(pool['sinusoid'], pool['original_novelty_curve'],
    #                   pool['framerate'], bpm, pool['length'])

    print 'bpms:', pool['peaksBpm']
    print 'first estimated bpms:', pool['first_estimated_bpms']
    if step>1:
        ticks = essentia.array(map(lambda i: ticks[i],
                               filter(lambda i: i%step == 0,range(len(ticks)))))

    pool.remove('ticks')
    pool.set('ticks', ticks)

def longestChain(dticks, startpos, period, tolerance):
    pos = startpos
    ubound = period*(1+tolerance)
    lbound = period*(1-tolerance)
    while (pos < len(dticks)) and\
          (lbound < dticks[pos] and dticks[pos] < ubound):
          pos += 1
    return pos - startpos

def alignTicks(sine, novelty, frameRate, bpm, size):
        ''' Aligns the sine function with the novelty function. Parameters:
            @sine: the sinusoid from bpmHistogram,
            @novelty: the novelty curve
            @frameRate: the frameRate
            @size: the audio size, in order to not to have more ticks than audiosize
            @bpm: the estimated bpm'''

        #pyplot.plot(novelty, 'k')
        #pyplot.plot(sine, 'r')
        #for i in range(len(novelty)-1):
        #    diff = novelty[i+1]-novelty[i]
        #    if diff > 0: novelty[i] = diff
        #    else: novelty[i] = 0
        #pyplot.plot(novelty, 'r')

        noveltySize = len(novelty)
        prodPulse = zeros(noveltySize, dtype='f4')
        i = 0
        while i < noveltySize:
            if sine[i] <= 0:
                i += 1
                continue
            window = []
            while i < noveltySize and sine[i] != 0:
              window.append(novelty[i]*sine[i])
              i+=1
            peakPos = argmax(window)
            peakPos = i - len(window) + peakPos
            prodPulse[peakPos] = novelty[peakPos]

        #pyplot.plot(prodPulse, 'g')
        #pyplot.show()
        ticks = []
        ticksAmp = []
        tatum = 60./bpm
        diffTick = 2*tatum
        prevTick = -1
        prevAmp = -1
        for i, x in enumerate(prodPulse):
            if x != 0:
               newTick = float(i)/frameRate
               if newTick < 0 or newTick >= size:
                   continue
               ticks.append(newTick)
               ticksAmp.append(x)
            #if x != 0:
            #    newTick = float(i)/frameRate
            #    if newTick < 0 or newTick >= size: continue
            #    if prevTick < 0:
            #       ticks.append(newTick)
            #       ticksAmp.append(x)
            #       prevTick = newTick
            #       prevAmp = x
            #    else:
            #        print 'ok'
            #        diff = newTick-prevTick
            #        if (diff >= 0.9*tatum) :
            #           ticks.append(newTick)
            #           ticksAmp.append(x)
            #           prevTick = newTick
            #           prevAmp = x
            #        else: #(newTick-prevTick) < 0.75*tatum:
            #            print 'newTick:', newTick, 'prevTick', prevTick, 'diff:', newTick-prevTick, 'tatum', tatum, 0.9*tatum
            #            newTick = (newTick*x+prevTick*prevAmp)/(x+prevAmp)
            #            ticks[-1] = newTick
            #            ticksAmp[-1] = (x+prevAmp)/2.
            #            prevTick = newTick
            #            prevAmp = (x+prevAmp)/2.
        return ticks, ticksAmp

def getMostStableTickLength(ticks):
    nticks = len(ticks)
    dticks = zeros(nticks-1)
    for i in range(nticks-1):
        dticks[i] = (ticks[i+1] - ticks[i])
    hist, distx = np.histogram(dticks, bins=50*(1+(max(dticks)-min(dticks))))
    bestPeriod = distx[argmax(hist)] # there may be more than one candidate!!
    bestBpm = 60./bestPeriod
    print 'best period', bestPeriod
    print 'best bpm:', bestBpm

    #print 'hist:', hist, distx
    maxLength = 0
    idx = 0
    for startpos in range(nticks-1):
        l = longestChain(dticks, startpos, bestPeriod, 0.1)
        if l > maxLength :
            maxLength = l;
            idx = startpos;

    print 'max stable length:', idx, maxLength
    return idx, maxLength, bestBpm



def postProcessTicks(audioFilename, ticks, ticksAmp, pool):
    '''Computes delta energy in order to find the correct position of the ticks'''
    # get rid of beats of beats > audio.length

    #    if t < 0 or t > pool['length']: continue
    #    ticks.append(float(t))
    #    ticksAmp.append(float(amp))

    #ticks = essentia.postProcessTicks(ticks, ticksAmp, 60./pool['harmonicBpm'][0]);
    beatWindowDuration = 0.01 # seconds
    beatDuration = 0.005      # seconds
    rmsFrameSize = 64
    rmsHopSize = rmsFrameSize/2
    audio = std.MonoLoader(filename=audioFilename,
                           sampleRate=pool['samplerate'],
                           downmix=pool['downmix'])()
    for i, tick in enumerate(ticks):
        startTime = tick - beatWindowDuration/2.0
        if startTime < 0: startTime = 0
        endTime = startTime + beatWindowDuration + beatDuration + 0.0001
        slice = std.Trimmer(sampleRate=pool['samplerate'],
                            startTime=startTime,
                            endTime=endTime)(audio)
        frames = std.FrameGenerator(slice, frameSize=rmsFrameSize, hopSize=rmsHopSize)
        maxDeltaRms=0
        RMS = std.RMS()
        prevRms = 0
        pos = 0
        tickPos = pos
        for frame in frames:
            rms = RMS(frame)
            diff = rms - prevRms
            if diff > maxDeltaRms:
                tickPos = pos
                maxDeltaRms = diff
            pos+=1
            prevRms = rms
        ticks[i]= tick + tickPos*float(rmsHopSize)/pool['samplerate']
    return ticks






def writeBeatFile(filename, pool) :
    beatFilename = os.path.splitext(filename)[0] + '_beat.wav' #'out_beat.wav' #
    audio = EasyLoader(filename=filename, downmix='mix', startTime=STARTTIME, endTime=ENDTIME)
    writer = MonoWriter(filename=beatFilename)
    onsetsMarker = AudioOnsetsMarker(onsets=pool['ticks'])
    audio.audio >> onsetsMarker.signal >> writer.audio
    essentia.run(audio)
    return beatFilename

def computeBeatsLoudness(filename, pool):
    loader = MonoLoader(filename=filename,
                        sampleRate=pool['samplerate'],
                        downmix=pool['downmix'])
    ticks = pool['ticks']#[pool['bestTicksStart']:pool['bestTicksStart']+32]
    beatsLoud = BeatsLoudness(sampleRate = pool['samplerate'],
                              frequencyBands = barkBands, #EqBands, #scheirerBands, #barkBands,
                              beats=ticks)
    loader.audio >> beatsLoud.signal
    beatsLoud.loudness >> (pool, 'loudness')
    beatsLoud.loudnessBandRatio >> (pool, 'loudnessBandRatio')
    essentia.run(loader)

def computeSpectrum(signal):
    #gen = VectorInput(signal)
    #fc = FrameCutter(startFromZero=False, frameSize=48, hopSize=1)
    #w = Windowing(zeroPhase=False)
    #spec = Spectrum()

    #p = essentia.Pool()
    #gen.data >> fc.signal
    #fc.frame >> w.frame >> spec.frame
    #spec.spectrum >> (p,'spectrum')
    #essentia.run(gen)

    #pyplot.imshow(p['spectrum'], cmap=pyplot.cm.hot, aspect='auto', origin='lower')

    corr = std.AutoCorrelation()(signal)
    pyplot.plot(corr)
    pyplot.show()
    print argmax(corr[2:])+2

def isPowerTwo(n):
    return (n&(n-1))==0

def isEvenHarmonic(a,b):
    if a < 2 or  b < 2: return False
    if (a<b): return isEvenHarmonic(b,a)
    return (a%b == 0) and isPowerTwo(a/b)

def isHarmonic(a,b):
    if a < 2 or  b < 2: return False
    if (a<b): return isHarmonic(b,a)
    return (a%b == 0)

def getHarmonics(array):
    size = len(array)
    hist = [0]*size
    counts = [1]*size
    for idx1, x in enumerate(array):
        for idx2, y in enumerate(array):
            if isEvenHarmonic(idx1, idx2):
                hist[idx1] += y
                counts[idx1] += 1
    hist = [hist[i]/float(counts[i]) for i in range(size)]
    return hist

def plot(pool, title, outputfile='out.svg', subplot=111):
    ''' plots bars for each beat'''

    #computeSpectrum(pool['loudness'])

    ticks = pool['ticks']
    #barSize = min([ticks[i+1] - ticks[i] for i in range(len(ticks[:-1]))])/2.
    barSize = 0.8
    offset = barSize/2.

    loudness = pool['loudness']
    loudnessBand = pool['loudnessBandRatio'] # ticks x bands

    medianRatiosPerTick = []
    meanRatiosPerTick = []
    for tick, energy in enumerate(loudnessBand):
            medianRatiosPerTick.append(median(energy))
            meanRatiosPerTick.append(mean(energy))


    loudnessBand = copy.deepcopy(loudnessBand.transpose()) # bands x ticks

    #xcorr = std.CrossCorrelation(minLag=0, maxLag=16)
    #acorr = std.AutoCorrelation()
    #bandCorr = []
    #for iBand, band in enumerate(loudnessBand):
    #    bandCorr.append(acorr(essentia.array(band)))

    nBands = len(loudnessBand)
    nticks = len(loudness)
    maxRatiosPerBand = []
    medianRatiosPerBand = []
    meanRatiosPerBand = []
    for idxBand, band in enumerate(loudnessBand):
        maxRatiosPerBand.append([0]*nticks)
        medianRatiosPerBand.append([0]*nticks)
        meanRatiosPerBand.append([0]*nticks)
        for idxTick in range(nticks):
            start = idxTick
            end = start+BEATWINDOW
            if (end>nticks):
                howmuch = end-nticks
                end = nticks-1
                start = end-howmuch
                if start < 0: start = 0
            medianRatiosPerBand[idxBand][idxTick] = median(band[start:end])
            maxRatiosPerBand[idxBand][idxTick] = max(band[start:end])
            meanRatiosPerBand[idxBand][idxTick] = mean(band[start:end])


    for iBand, band in enumerate(loudnessBand):
        for tick, ratio in enumerate(band):
            #if ratio < medianRatiosPerBand[iBand][tick] and\
            #   ratio <= medianRatiosPerTick[tick]: loudnessBand[iBand][tick]=0
            bandThreshold = max(medianRatiosPerBand[iBand][tick],
                                meanRatiosPerBand[iBand][tick])
            tickThreshold = max(medianRatiosPerTick[tick],
                                meanRatiosPerTick[tick])
            if ratio < bandThreshold and ratio <= tickThreshold:
                loudnessBand[iBand][tick]=0
            else:
                loudnessBand[iBand][tick] *= loudness[tick]
                #if loudnessBand[iBand][tick] > 1 : loudnessBand[iBand][tick] = 1

    acorr = std.AutoCorrelation()
    bandCorr = []
    maxCorr = []
    for iBand, band in enumerate(loudnessBand):
        bandCorr.append(acorr(essentia.array(band)))
        maxCorr.append(argmax(bandCorr[-1][2:])+2)

    # use as much window space as possible:
    pyplot.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    pyplot.subplot(511)
    pyplot.imshow(bandCorr, cmap=pyplot.cm.hot, aspect='auto', origin='lower', interpolation='nearest')
    print 'max correlation', maxCorr

    sumCorr = []
    for tick in range(nticks):
        total = 0
        for band in bandCorr:
            total += band[tick]
        sumCorr.append(total)

    sumCorr[0] = 0
    sumCorr[1] = 0
    pyplot.subplot(512)
    maxAlpha = max(sumCorr)
    for i,val in enumerate(sumCorr):
        alpha = max(0,min(val/maxAlpha, 1))
        pyplot.bar(i, 1 , barSize, align='edge',
                   bottom=0,alpha=alpha,
                   color='r', edgecolor='w', linewidth=.3)

    print 'max sum correlation', argmax(sumCorr[2:])+2

    hist = getHarmonics(sumCorr)
    maxHist = argmax(hist)
    print 'max histogram', maxHist
    #for idx,val in enumerate(hist):
    #    if val < maxHist: hist[idx] = 0

    pyplot.subplot(513)
    for i,val in enumerate(hist):
        pyplot.bar(i, val , barSize, align='edge',
                   bottom=0, color='r', edgecolor='w', linewidth=.3)


    peakDetect = std.PeakDetection(maxPeaks=5,
                                   orderBy='amplitude',
                                   minPosition=0,
                                   maxPosition=len(sumCorr)-1,
                                   range=len(sumCorr)-1)
    peaks = peakDetect(sumCorr)[0]
    peaks = [round(x+1e-15) for x in peaks]
    print 'Peaks:',peaks

    pyplot.subplot(514)
    maxAlpha = max(sumCorr)
    for i,val in enumerate(sumCorr):
        alpha = max(0,min(val/maxAlpha, 1))
        pyplot.bar(i, val, barSize, align='edge',
                   bottom=0,alpha=alpha,
                   color='r', edgecolor='w', linewidth=.3)

    # multiply both histogram and sum corr to have a weighted histogram:
    wHist = essentia.array(hist)*sumCorr*acorr(loudness)
    maxHist = argmax(wHist)
    print 'max weighted histogram', maxHist
    pyplot.subplot(515)

    maxAlpha = max(wHist)
    for i,val in enumerate(wHist):
        alpha = max(0,min(val/maxAlpha, 1))
        pyplot.bar(i, val, barSize, align='edge',
                   bottom=0,alpha=alpha,
                   color='r', edgecolor='w', linewidth=.3)

    pyplot.savefig(outputfile, dpi=300)
    #pyplot.show()
    return



def ossplay(filename): # play audio thru oss
    from wave import open as waveOpen
    from ossaudiodev import open as ossOpen
    s = waveOpen(filename,'rb')
    (nc,sw,fr,nf,comptype, compname) = s.getparams( )
    dsp = ossOpen('/dev/dsp','w')
    try:
        from ossaudiodev import AFMT_S16_NE
    except ImportError:
        if byteorder == "little":
            AFMT_S16_NE = ossaudiodev.AFMT_S16_LE
        else:
            AFMT_S16_NE = ossaudiodev.AFMT_S16_BE
    dsp.setparameters(AFMT_S16_NE, nc, fr)
    data = s.readframes(nf)
    s.close()
    dsp.write(data)
    dsp.close()

def getkey(audioFilename, device, f, card, lock):
    c = None
    b = True
    while b:
        #fd = sys.stdin.fileno()
        #old = termios.tcgetattr(fd)
        #new = termios.tcgetattr(fd)
        #new[3] = new[3] & ~TERMIOS.ICANON & ~TERMIOS.ECHO
        #new[6][TERMIOS.VMIN] = 1
        #new[6][TERMIOS.VTIME] = 0
        #termios.tcsetattr(fd, TERMIOS.TCSANOW, new)
        #c = None
        lock.acquire()
        #try:
        #        c = os.read(fd, 1)
        #finally:
        #        termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
        #if c == '\n':     ## break on a Return/Enter keypress
        #   b = False
        #   return
        #if c==' ': playAudio(audioFilename)
        #else: print 'got', c
        #ossplay(audioFilename)
        alsaplay(audioFilename, device, f, card)
        lock.release()
        time.sleep(0.1)

def alsaplay(filename, device, f, card):
    device.setchannels(f.getnchannels())
    device.setrate(f.getframerate())

    # 8bit is unsigned in wav files
    if f.getsampwidth() == 1:
        device.setformat(alsaaudio.PCM_FORMAT_U8)
    # Otherwise we assume signed data, little endian
    elif f.getsampwidth() == 2:
        device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    elif f.getsampwidth() == 3:
        device.setformat(alsaaudio.PCM_FORMAT_S24_LE)
    elif f.getsampwidth() == 4:
        device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    else:
        raise ValueError('Unsupported format')

    device.setperiodsize(320)

    data = f.readframes(320)
    while data:
        device.write(data)
        data = f.readframes(320)
    f.setpos(0)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        usage()
        sys.exit(1)
    step = 1
    if len(sys.argv) > 2:
        step = int(sys.argv[-1])
    inputfilename = sys.argv[1]
    ext = os.path.splitext(inputfilename)[1]
    if ext == '.txt': # input file contains a list of audio files
        files = open(inputfilename).read().split('\n')[:-1]
    else: files = [inputfilename]

    for audiofile in files:
        print "*"*70
        print "Processing ", audiofile
        print "*"*70
        try:
            bpmfile = audiofile.replace('wav', 'bpm')
            print "bpmfile:", bpmfile
            print 'realBpm', open(bpmfile).read()
        except:
            print 'realBpm not found'

        pool = essentia.Pool()
        pool.set('downmix',    DOWNMIX)
        pool.set('framesize',  FRAMESIZE)
        pool.set('hopsize',    HOPSIZE)
        pool.set('weight',     WEIGHT)
        pool.set('samplerate', SAMPLERATE)
        pool.set('window',     WINDOW)
        pool.set('framerate',  FRAMERATE)
        pool.set('tempo_framesize', TEMPO_FRAMESIZE)
        pool.set('tempo_overlap',   TEMPO_OVERLAP)
        pool.set('step', step)

        #computeSegmentation(audiofile, pool)
        #segments = pool['segments']
        computeBeats(audiofile, pool)
        beatFilename = writeBeatFile(audiofile, pool)
        computeBeatsLoudness(audiofile, pool)

        imgfilename = os.path.splitext(audiofile)[0]+'.png'
        #imgfilename = imgfilename.split(os.sep)[-1]
        #print 'plotting', imgfilename

        if sys.platform == 'darwin' or sys.platform == 'win32':
            plot(pool,'beats loudness ' + str(audiofile), imgfilename);
        else:
           # card = 'default'
           # f = wave.open(beatFilename, 'rb')
           ## print '%d channels, sampling rate: %d \n' % (f.getnchannels(),
           ##                                            f.getframerate())
           # device = alsaaudio.PCM(card=card)

           # lock = thread.allocate_lock()
           # thread.start_new_thread(getkey, (beatFilename, device, f, card, lock))


            plot(pool,'beats loudness ' + audiofile, imgfilename);
           # f.close()
           # thread.exit()

        #print 'deleting beatfile:', beatFilename
        #subprocess.call(['rm', beatFilename])
