import sys
import essentia.standard as es
from essentia import *
import numpy
import pylab


try:
    input_file = sys.argv[1]
except:
    print ("usage: %s <input_file>" % sys.argv[0])
    sys.exit()


frameSize = 2048
hopSize = 128
weight = 'hybrid'

print("Frame size: %d" % frameSize)
print("Hop size: %d" % hopSize)
print("weight: %s" % weight)

audio = es.MonoLoader(filename=input_file)()

w = es.Windowing(type='hann')
s = es.Spectrum()
freq_bands = es.FrequencyBands()

bands_energies = []
for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    bands_energies.append(freq_bands(s(w(frame))))

novelty = es.NoveltyCurve(frameRate=44100./hopSize, weightCurveType=weight)(numpy.array(bands_energies))
bpm, candidates, magnitudes, tempogram, _, ticks, ticks_strength, sinusoid = es.BpmHistogram(frameRate=44100./hopSize)(novelty)

print("BPM = %0.1f" % bpm)
   
#pylab.plot(novelty)
#pylab.show()
pylab.matshow(tempogram.transpose(), origin='lower', aspect='auto')
pylab.show()
