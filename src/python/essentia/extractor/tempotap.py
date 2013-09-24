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

#! /usr/bin/python

import essentia
from essentia import INFO, isSilent

from numpy import bincount, argmax, mean

namespace = 'rhythm'
dependencies = None


def lagtobpm(lag, sampleRate, hopsize):
    return 60.0 * sampleRate / lag / hopsize

def compute(audio, pool, options):

    INFO('Computing Tempo extractor...')

    use_onset   = options['useOnset']
    use_bands   = options['useBands']

    # frameNumber * hopSize ~= about 6 seconds
    hopSize     = options['hopSize']
    frameSize   = options['frameSize']
    frameNumber = options['frameNumber']
    frameHop    = options['frameHop']
    sampleRate  = options['sampleRate']
    tolerance   = 0.24
    period_tol  = 2
    windowType  = options['windowType']

    bands_freq = [40.0, 413.16, 974.51, 1818.94, 3089.19, 5000.0, 7874.4, 12198.29, 17181.13]
    bands_gain = [2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5]
    maxbpm = 208
    minbpm = 40
    last_beat_interval = 0.025
    frame_time = float(hopSize) / float(sampleRate)

    frames           = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window           = essentia.Windowing(size = frameSize, zeroPadding = 0, type = windowType)
    if use_onset:
        fft              = essentia.FFT(size = frameSize)
        cartesian2polar  = essentia.CartesianToPolar()
        onset_hfc        = essentia.OnsetDetection(method = 'hfc', sampleRate = sampleRate)
        onset_complex    = essentia.OnsetDetection(method = 'complex', sampleRate = sampleRate)
    if use_bands:
        espectrum        = essentia.Spectrum(size = frameSize)
        tempotapbands    = essentia.FrequencyBands(frequencyBands = bands_freq)
        temposcalebands  = essentia.TempoScaleBands(bandsGain = bands_gain)
    tempotap         = essentia.TempoTap(numberFrames = frameNumber, sampleRate = sampleRate, frameHop = frameHop)
    tempotapticks    = essentia.TempoTapTicks(hopSize = hopSize, sampleRate = sampleRate, frameHop = frameHop)

    frameTime = float(hopSize) / float(sampleRate)
    frameRate = 1. / frameTime

    nframes = 0
    bpm_estimates_list = []
    ticks = []
    matchingPeriods = []
    oldhfc = 0

    fileLength = len(audio)/sampleRate
    startSilence = 0
    oldSilence = 0
    endSilence = round(fileLength * sampleRate / hopSize) + 1

    for frame in frames:
        windowed_frame = window(frame)
        features = []
        if use_onset:
            complex_fft = fft(windowed_frame)
            (spectrum,phase) = cartesian2polar(complex_fft)
            hfc = onset_hfc(spectrum,phase)
            complexdomain = onset_complex(spectrum,phase)
            difhfc = max(hfc - oldhfc,0)
            oldhfc = hfc
            features += [hfc,difhfc,complexdomain]
        if use_bands:
            spectrum_frame = espectrum(windowed_frame)
            bands = tempotapbands(spectrum_frame)
            (scaled_bands, cumul) = temposcalebands(bands)
            features += list(scaled_bands)

        features = essentia.array(features)
        (periods, phases) = tempotap(features)
        (these_ticks, these_matchingPeriods) = tempotapticks(periods, phases)
        for period in these_matchingPeriods:
          if period != 0:
            matchingPeriods += [ period ]
        ticks += list(these_ticks)

        if nframes < 5. * sampleRate / hopSize:
          if isSilent(frame) and startSilence == nframes - 1:
            startSilence = nframes

        if nframes > (fileLength - 5.) * sampleRate / hopSize:
          if isSilent(frame):
            if oldSilence != nframes - 1:
              endSilence = nframes
            oldSilence = nframes

        nframes += 1

    # make sure we do not kill beat too close to music
    if startSilence > 0: startSilence -= 1
    endSilence += 1

    # fill the rest of buffer with zeros
    features = essentia.array([0]*len(features))
    while nframes % frameNumber != 0:
        (periods, phases) = tempotap(features)
        (these_ticks, these_matchingPeriods) = tempotapticks(periods, phases)
        ticks += list(these_ticks)
        matchingPeriods += list(these_matchingPeriods)
        nframes += 1

    if len(ticks) > 2:
      # fill up to end of file
      if fileLength > ticks[-1]:
        lastPeriod = ticks[-1] - ticks[-2]
        while ticks[-1] + lastPeriod < fileLength - last_beat_interval:
          if ticks[-1] > fileLength - last_beat_interval:
            break
          ticks.append(ticks[-1] + lastPeriod)
    if len(ticks) > 1:
      # remove all negative ticks
      i = 0
      while i < len(ticks):
        if ticks[i] < startSilence / sampleRate * hopSize: ticks.pop(i)
        else: i += 1
      # kill all ticks from 350ms before the end of the song
      i = 0
      while i < len(ticks):
        if ticks[i] > endSilence / sampleRate * hopSize: ticks.pop(i)
        else: i += 1
      # prune values closer than tolerance
      i = 1
      while i < len(ticks):
        if ticks[i] - ticks[i-1] < tolerance: ticks.pop(i)
        else: i += 1
      # prune all backward offbeat
      i = 3
      while i < len(ticks):
        if    abs( (ticks[i] - ticks[i-2]) - 1.5 * (ticks[i]   - ticks[i-1]) ) < 0.100 \
          and abs( (ticks[i] - ticks[i-1]) -       (ticks[i-2] - ticks[i-3]) ) < 0.100 :
          ticks.pop(i-2)
        else: i += 1


    for period in matchingPeriods:
      if period != 0:
        bpm_estimates_list += [ lagtobpm(period, sampleRate, hopSize) ]
      #else:
      #  bpm_estimates_list += [ 0 ]

    # bpm estimates
    for bpm_estimate in bpm_estimates_list:
        pool.add(namespace + '.' + 'bpm_estimates', bpm_estimate)

    # estimate the bpm from the list of candidates
    if len(bpm_estimates_list) > 0:
      estimates = [bpm/2. for bpm in bpm_estimates_list]
      closestBpm = argmax(bincount(estimates))*2.
      matching = []
      for bpm in bpm_estimates_list:
        if abs(closestBpm - bpm) < period_tol:
          matching.append(bpm)
      if (len(matching) < 1):
        # something odd happened
        bpm = closestBpm
      else :
        bpm = mean(matching)
    else:
      bpm = 0.
    # convert to floats, as python bindings yet not support numpy.float32
    ticks = [float(tick) for tick in ticks]
    pool.add(namespace + '.' + 'bpm', bpm)#, pool.GlobalScope)
    pool.add(namespace + '.' + 'beats_position', ticks)#, pool.GlobalScope

    bpm_intervals = [ticks[i] - ticks[i-1] for i in range(1, len(ticks))]
    pool.add(namespace + '.' + 'bpm_intervals', bpm_intervals)#, pool.GlobalScope

    from numpy import histogram
    tempotap_bpms = [60./i for i in bpm_intervals]
    if len(tempotap_bpms) > 0:
      weight, values = histogram(tempotap_bpms, bins = 250, range = (0,250), normed=True)
    else:
      weight, values = [0.], [0.]
    first_peak_weights = [0] * 250
    secnd_peak_weights = [0] * 250

    for i in range(max(argmax(weight)-4,0), min(argmax(weight)+5,len(weight)) ):
      first_peak_weights[i] = weight[i]
      weight[i] = 0.
    for i in range(max(argmax(weight)-4,0), min(argmax(weight)+5,len(weight)) ):
      secnd_peak_weights[i] = weight[i]
      weight[i] = 0.

    pool.add(namespace + '.' + 'first_peak_bpm', values[argmax(first_peak_weights)])#, pool.GlobalScope
    pool.add(namespace + '.' + 'first_peak_weight', first_peak_weights[argmax(first_peak_weights)])#, pool.GlobalScope
    if sum(first_peak_weights) != 0.:
      pool.add(namespace + '.' + 'first_peak_spread', 1.-first_peak_weights[argmax(first_peak_weights)]/sum(first_peak_weights))#, pool.GlobalScope
    else:
      pool.add(namespace + '.' + 'first_peak_spread', 0.)#, pool.GlobalScope
    pool.add(namespace + '.' + 'second_peak_bpm', values[argmax(secnd_peak_weights)])#, pool.GlobalScope
    pool.add(namespace + '.' + 'second_peak_weight', secnd_peak_weights[argmax(secnd_peak_weights)])#, pool.GlobalScope
    if sum(secnd_peak_weights) != 0.:
      pool.add(namespace + '.' + 'second_peak_spread', 1.-secnd_peak_weights[argmax(secnd_peak_weights)]/sum(secnd_peak_weights))#, pool.GlobalScope
    else:
      pool.add(namespace + '.' + 'second_peak_spread', 0.)#, pool.GlobalScope

    '''
    def rubato(ticks):
        bpm_rubato_python = []
        tolerance = 0.08
        i = 5
        tmp1 = 60./ float(ticks[i  ] - ticks[i-1])
        tmp2 = 60./ float(ticks[i-1] - ticks[i-2])
        tmp3 = 60./ float(ticks[i-2] - ticks[i-3])
        tmp4 = 60./ float(ticks[i-3] - ticks[i-4])
        tmp5 = 60./ float(ticks[i-4] - ticks[i-5])
        for i in range(6, len(ticks)):
            if (  abs(1. - tmp1 / tmp4) >= tolerance
              and abs(1. - tmp2 / tmp5) >= tolerance
              and abs(1. - tmp2 / tmp4) >= tolerance
              and abs(1. - tmp1 / tmp5) >= tolerance
              and abs(1. - tmp1 / tmp2) <= tolerance
              and abs(1. - tmp4 / tmp5) <= tolerance ):
                bpm_rubato_python.append(ticks[i-2])
            tmp5 = tmp4; tmp4 = tmp3; tmp3 = tmp2; tmp2 = tmp1
            tmp1 = 60./ (ticks[i] - ticks[i-1])
        print bpm_rubato_python
        return bpm_rubato_python
    '''
    # FIXME we need better rubato algorithm
    #rubato = essentia.BpmRubato()
    #bpm_rubato_start, bpm_rubato_stop = rubato(ticks)
    #pool.add(namespace + '.' + 'rubato_start', bpm_rubato_start)#, pool.GlobalScope
    #pool.add(namespace + '.' + 'rubato_stop',  bpm_rubato_stop)#,  pool.GlobalScope)

    INFO('100% done...')
