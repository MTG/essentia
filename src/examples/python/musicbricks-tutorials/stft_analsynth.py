# import essentia in standard mode

#import essentia.standard
#from essentia.standard import *

import essentia
import essentia.streaming as es
import essentia.standard as std


# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import sys

mode = 'streaming' #'standard' #'streaming'


if len(sys.argv)<3:
  print ("analysis / synthesis STFT example")
  print ("Usage: stft_analsynth.py in_wav out_wav [mode]")
  print("mode can be: streaming or standard")
  exit()
else:
  inputFilename = sys.argv[1]
  outputFilename = sys.argv[2]
  
if len(sys.argv) == 4:
  mode = sys.argv[3]


# algorithm parameters
framesize = 1024
hopsize = 256


# loop over all frames
audioout = np.array(0)
counter = 0


import matplotlib.pylab as plt

if mode == 'standard':

  # create an audio loader and import audio file
  loader = std.MonoLoader(filename = inputFilename, sampleRate = 44100)
  audio = loader()

  print("Duration of the audio sample [sec]:")
  print(len(audio)/44100.0)

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

    if counter >= (framesize/(2*hopsize)):
      audioout = np.append(audioout, out)
    counter += 1

  # write audio output
  print audioout.shape
  awrite(audioout.astype(np.float32))


if mode == 'streaming':
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
  loader.audio >> fcut.signal
  fcut.frame >> w.frame
  w.frame >> fft.frame
  fft.fft >> ifft.fft
  ifft.frame >> overl.frame
  overl.signal >> awrite.audio
  overl.signal >> (pool, 'audio')
  
  
  essentia.run(loader)

  print type(pool['audio'])
  print pool['audio'].shape





