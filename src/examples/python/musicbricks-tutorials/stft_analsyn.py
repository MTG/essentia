# import essentia in standard mode
import essentia
import essentia.standard
from essentia.standard import *

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

# audio filename
inputFilename = 'predom.wav'
outputFilename = 'predom_stft.wav'


# algorithm parameters
framesize = 2048
hopsize = 256

# create an audio loader and import audio file
loader = essentia.standard.MonoLoader(filename = inputFilename, sampleRate = 44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/44100.0)


fcut = FrameCutter(frameSize = framesize, hopSize = hopsize);
w = Windowing(type = "hann");
fft = FFT(size = framesize);
ifft = IFFT(size = framesize);
overl = OverlapAdd (frameSize = framesize, hopSize = hopsize);
awrite = MonoWriter (filename = outputFilename, sampleRate = 44100);


# loop over all frames
audioout = np.array(0)

for frame in FrameGenerator(audio, frameSize = framesize, hopSize = hopsize):
    infft = fft(w(frame))
    outfft = infft
    out = overl(ifft(outfft))
    audioout = np.append(audioout, out)

# write audio output
awrite(audio)



