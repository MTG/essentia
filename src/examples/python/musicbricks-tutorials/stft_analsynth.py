# import essentia in standard mode

#import essentia.standard
#from essentia.standard import *

import essentia
import essentia.streaming as es
import essentia.standard as std


# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np


# audio filename
inputFilename = '/Users/jjaner/VoctroLabs/Projects/MusicMastermind/test-perfanalysis/sine_12_6dB.wav' # 'predom.wav'
outputFilename = 'out_stft.wav'


# algorithm parameters
framesize = 1024
hopsize = 256
mode = 'standard_mode'#''streaming_mode'


# loop over all frames
audioout = np.array(0)
counter = 0


import matplotlib.pylab as plt

if mode == 'standard_mode':

  # create an audio loader and import audio file
  loader = std.MonoLoader(filename = inputFilename, sampleRate = 44100)
  audio = loader()

  print("Duration of the audio sample [sec]:")
  print(len(audio)/44100.0)

  fcut = std.FrameCutter(frameSize = framesize, hopSize = hopsize, startFromZero =  False);
  w = std.Windowing(type = "hann");
  fft = std.FFT(size = framesize);
  ifft = std.IFFT(size = framesize);
  overl = std.OverlapAdd (frameSize = framesize, hopSize = hopsize);
  awrite = std.MonoWriter (filename = outputFilename, sampleRate = 44100);


  for frame in std.FrameGenerator(audio, frameSize = framesize, hopSize = hopsize):
    # STFT analysis
    infft = fft(w(frame))
    
    # here we could apply spectral transformations
    outfft = infft

    # STFT synthesis
    ifftframe = ifft(outfft)
    out = overl(ifftframe)
    
#    print out.shape
#    #plt debug
#    plt.subplot(3,1,1)
#    plt.plot(w(frame))
#    plt.subplot(3,1,2)
#    plt.plot(ifftframe,'r')
#    plt.subplot(3,1,3)
#    plt.plot(out,'g')
#    plt.show()
#    if counter > 10:
#      break;

    if counter >= (framesize/(2*hopsize)):
      audioout = np.append(audioout, out)
    counter += 1

  # write audio output
  print audioout.shape
  awrite(audioout.astype(np.float32))


if mode == 'streaming_mode':
  out = np.array(0)
  loader = es.MonoLoader(filename = inputFilename, sampleRate = 44100)
  pool = essentia.Pool()
  fcut = es.FrameCutter(frameSize = framesize, hopSize = hopsize, startFromZero =  False);
  w = es.Windowing(type = "hann");
  fft = es.FFT(size = framesize);
  ifft = es.IFFT(size = framesize);
  overl = es.OverlapAdd (frameSize = framesize, hopSize = hopsize);
  awrite = es.MonoWriter (filename = outputFilename, sampleRate = 44100);
  
  #gen = audio #VectorInput(audio)
  #  pool = Pool()
  loader.audio >> fcut.signal
  fcut.frame >> w.frame
  w.frame >> fft.frame
  fft.fft >> ifft.fft
  ifft.frame >> overl.frame
  overl.signal >> (pool, 'audio')
  overl.signal >> awrite.audio
  
  
  
  essentia.run(loader)
#audioout = np.append(audioout, out)







