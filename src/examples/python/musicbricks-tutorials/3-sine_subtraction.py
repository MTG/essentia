
import essentia
import essentia.streaming as es

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import sys

# algorithm parameters
params = { 'frameSize': 2048, 'hopSize': 128, 'startFromZero': False, 'sampleRate': 44100,'maxnSines': 100,'magnitudeThreshold': -74,'minSineDur': 0.02,'freqDevOffset': 10, 'freqDevSlope': 0.001}


# loop over all frames
audioout = np.array(0)
counter = 0

# input and output files
import os.path
tutorial_dir = os.path.dirname(os.path.realpath(__file__))
inputFilename = os.path.join(tutorial_dir, 'singing-female.wav')
outputFilename = os.path.join(tutorial_dir, 'singing-female-out-sinesubtraction.wav')


out = np.array(0)
loader = es.MonoLoader(filename = inputFilename, sampleRate =  params['sampleRate'])
pool = essentia.Pool()
fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False);
w = es.Windowing(type = "blackmanharris92");
fft = es.FFT(size = params['frameSize']);
smanal = es.SineModelAnal(sampleRate = params['sampleRate'], maxnSines = params['maxnSines'], magnitudeThreshold = params['magnitudeThreshold'], freqDevOffset = params['freqDevOffset'], freqDevSlope = params['freqDevSlope'])
subtrFFTSize = min(params['frameSize']/4, 4* params['hopSize'])
smsub = es.SineSubtraction(sampleRate = params['sampleRate'], fftSize = subtrFFTSize, hopSize = params['hopSize'])


# analysis
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> smanal.fft
smanal.magnitudes >> (pool, 'magnitudes')
smanal.frequencies >> (pool, 'frequencies')
smanal.phases >> (pool, 'phases')
# subtraction
fcut.frame >> smsub.frame
smanal.magnitudes >> smsub.magnitudes
smanal.frequencies >> smsub.frequencies
smanal.phases >> smsub.phases
smsub.frame >> (pool, 'frames')


essentia.run(loader)


#store to file
outaudio = pool['frames'].flatten() 

awrite = es.MonoWriter (filename = outputFilename, sampleRate =  params['sampleRate']);
outvector = es.VectorInput(outaudio)

outvector.data >> awrite.audio
essentia.run(outvector)


