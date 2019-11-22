
import essentia
import essentia.streaming as es

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import sys

# algorithm parameters
params = { 'frameSize': 2048, 'hopSize': 512, 'startFromZero': False, 'sampleRate': 44100,'maxnSines': 100,'magnitudeThreshold': -74,'minSineDur': 0.02,'freqDevOffset': 10, 'freqDevSlope': 0.001}


# loop over all frames
audioout = np.array(0)
counter = 0

# input and output files
import os.path
tutorial_dir = os.path.dirname(os.path.realpath(__file__))
inputFilename = os.path.join(tutorial_dir, 'singing-female.wav')
outputFilename = os.path.join(tutorial_dir, 'singing-female-out-sinemodel.wav')

out = np.array(0)
loader = es.MonoLoader(filename = inputFilename, sampleRate =  params['sampleRate'])
pool = essentia.Pool()
fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False);
w = es.Windowing(type = "blackmanharris92");
fft = es.FFT(size = params['frameSize']);
smanal = es.SineModelAnal(sampleRate = params['sampleRate'], maxnSines = params['maxnSines'], magnitudeThreshold = params['magnitudeThreshold'], freqDevOffset = params['freqDevOffset'], freqDevSlope = params['freqDevSlope'])
smsyn = es.SineModelSynth(sampleRate = params['sampleRate'], fftSize = params['frameSize'], hopSize = params['hopSize'])
ifft = es.IFFT(size = params['frameSize']);
overl = es.OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize'] );
awrite = es.MonoWriter (filename = outputFilename, sampleRate =  params['sampleRate']);



# analysis
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> smanal.fft
smanal.magnitudes >> (pool, 'magnitudes')
smanal.frequencies >> (pool, 'frequencies')
smanal.phases >> (pool, 'phases')
# synthesis
smanal.magnitudes >> smsyn.magnitudes
smanal.frequencies >> smsyn.frequencies
smanal.phases >> smsyn.phases
smsyn.fft >> ifft.fft
ifft.frame >> overl.frame
overl.signal >> awrite.audio
overl.signal >> (pool, 'audio')




essentia.run(loader)






