
import essentia
import essentia.streaming as es

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import sys

# algorithm parameters
framesize = 1024
hopsize = 256

# loop over all frames
audioout = np.array(0)
counter = 0

import matplotlib.pylab as plt

# input and output files
import os.path
tutorial_dir = os.path.dirname(os.path.realpath(__file__))
inputFilename = os.path.join(tutorial_dir, 'singing-female.wav')
outputFilename = os.path.join(tutorial_dir, 'singing-female-out-stft.wav')

out = np.array(0)
loader = es.MonoLoader(filename = inputFilename, sampleRate = 44100)
pool = essentia.Pool()
fcut = es.FrameCutter(frameSize = framesize, hopSize = hopsize, startFromZero =  False);
w = es.Windowing(type = "hann");
fft = es.FFT(size = framesize);
ifft = es.IFFT(size = framesize);
overl = es.OverlapAdd (frameSize = framesize, hopSize = hopsize, gain = 1./framesize );
awrite = es.MonoWriter (filename = outputFilename, sampleRate = 44100);

loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> ifft.fft
ifft.frame >> overl.frame
overl.signal >> awrite.audio
overl.signal >> (pool, 'audio')

essentia.run(loader)






