# import essentia in standard mode
import essentia
import essentia.standard
from essentia.standard import *

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

# audio filename
inputFilename = '/Users/jjaner/VoctroLabs/Projects/MusicMastermind/test-perfanalysis/sine_12_6dB.wav' # 'predom.wav'
outputFilename = 'out_stft.wav'


# algorithm parameters
framesize = 1024
hopsize = 256

# create an audio loader and import audio file
loader = essentia.standard.MonoLoader(filename = inputFilename, sampleRate = 44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/44100.0)


fcut = FrameCutter(frameSize = framesize, hopSize = hopsize, startFromZero =  False);
w = Windowing(type = "hann");
fft = FFT(size = framesize);
ifft = IFFT(size = framesize);
overl = OverlapAdd (frameSize = framesize, hopSize = hopsize);
awrite = MonoWriter (filename = outputFilename, sampleRate = 44100);


# loop over all frames
audioout = np.array(0)
counter = 0

import matplotlib.pylab as plt

for frame in FrameGenerator(audio, frameSize = framesize, hopSize = hopsize):
    # STFT analysis
    infft = fft(w(frame))
    
    # here we could apply spectral transformations
    outfft = infft

    # STFT synthesis
    ifftframe = ifft(outfft)
    out = overl(ifftframe)
    print out.shape
    #plt debug
    plt.subplot(3,1,1)
    plt.plot(w(frame))
    plt.subplot(3,1,2)
    plt.plot(ifftframe,'r')
    plt.subplot(3,1,3)
    plt.plot(out,'g')
    plt.show()
    if counter > 10:
      break;

    if counter >= (framesize/(2*hopsize)):
      audioout = np.append(audioout, out)
    counter += 1

# write audio output
print audioout.shape
awrite(audioout.astype(np.float32))



