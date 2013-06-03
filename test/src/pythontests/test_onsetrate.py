#!/usr/bin/env python
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
