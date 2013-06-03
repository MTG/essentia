#!/usr/bin/env python
import sys
import essentia
import os
import glob
import numpy

sample_rate = 44100.0
frame_size = 32768
hop_size = 16384
frame_rate = (frame_size - hop_size) / sample_rate
zero_padding = 0

all_files = glob.glob('../../../test/audio/patterns/*.mp3')

for input_file in all_files:

    audio   = essentia.AudioFileInput(filename = input_file)
    samples = audio()
    frames  = essentia.FrameGenerator(audio = samples, frameSize = frame_size, hopSize = hop_size)
    window  = essentia.Windowing(windowSize = frame_size, zeroPadding = zero_padding, type = "blackmanharris62")
    spectrum = essentia.Spectrum(size = frame_size + zero_padding)
    spectralpeaks = essentia.SpectralPeaks(maxPeaks = 100, magnitudeThreshold = -100, minFrequency = 130, maxFrequency = 8000, highestPeaks = True)
    lin2db = essentia.UnaryOperator(type = "lin2db")
    db2lin = essentia.UnaryOperator(type = "db2lin")
    hpcp = essentia.HPCP(size = 12, referenceFrequency = 440, useWeight = True, windowSize = 4.0/3.0, sampleRate = sample_rate, normalize = True)
    patterndetection = essentia.PatternDetection(frameRate = frame_rate, lengthMin = 10.0, lengthMax = 15.0)
    total_frames = frames.num_frames()
    n_frames = 0
    
    for frame in frames:
        
        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)
        (frame_peaks_freq, frame_peaks_mag) = spectralpeaks(lin2db(frame_spectrum))

        if len(frame_peaks_freq) and len(frame_peaks_mag):
            frame_peaks_mag = db2lin(frame_peaks_mag)
            frame_hpcp = hpcp(frame_peaks_freq, frame_peaks_mag)

        try:
            chroma = numpy.concatenate([chroma,
                                        essentia.array([frame_hpcp], ndmin=2) ],
                                        axis = 0)
        except:
            chroma = essentia.array([frame_hpcp], ndmin=2)

        n_frames += 1

    (r, rSegments, patternOnsets, patternRate) = patterndetection(chroma);
   
    print patternRate , os.path.basename(input_file)
