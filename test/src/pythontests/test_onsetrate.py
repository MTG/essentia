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
import essentia
import os
import glob
import numpy

sample_rate = 44100.0
frame_size = 1024
hop_size = 512
frame_rate = (frame_size - hop_size) / sample_rate
zero_padding = 0

for input_file in glob.glob('../../../../audio/recorded/*.wav'):
    audio   = essentia.AudioFileInput(filename = input_file)
    samples = audio()
    frames  = essentia.FrameGenerator(audio = samples, frameSize = frame_size, hopSize = hop_size)
    window  = essentia.Windowing(windowSize = frame_size, zeroPadding = zero_padding, type = "hann")
    fft     = essentia.FFT()
    cartesian2polar = essentia.Cartesian2Polar()
    onsetdetectionHFC = essentia.OnsetDetection(method = "hfc", sampleRate = sample_rate)
    onsetdetectionComplex = essentia.OnsetDetection(method = "complex", sampleRate = sample_rate)
    onsets = essentia.Onsets(frameRate = frame_rate, alpha = 0.2, delayCoef = 6, silenceTS = 0.075)

    total_frames = frames.num_frames()
    n_frames = 0

    hfc = []
    complex = []

    for frame in frames:

        windowed_frame = window(frame)
        complex_fft = fft(windowed_frame)
        (spectrum,phase) = cartesian2polar(complex_fft)
        hfc.append(onsetdetectionHFC(spectrum,phase))
        complex.append(onsetdetectionComplex(spectrum,phase))
        n_frames += 1

    detections = numpy.concatenate([essentia.array([hfc]),
                                    essentia.array([complex]) ])
    time_onsets = onsets(detections, essentia.array([1, 1]))

    print len(time_onsets) / ( len(samples) / sample_rate ), os.path.basename(input_file)
