from configparser import Interpolation
import sys
import essentia.standard as es
import numpy
import pylab


try:
    input_file = sys.argv[1]
except:
    print("usage: {sys.argv[0]} <input_file>")
    sys.exit()

sample_rate = 44100
frame_size = 2048
hop_size = 128
weight = "hybrid"

print(f"Sample rate: {sample_rate}")
print(f"Frame size: {frame_size}")
print(f"Hop size: {hop_size}")
print(f"weight: {weight}")

audio = es.MonoLoader(filename=input_file)()

w = es.Windowing(type="hann")
s = es.Spectrum()
freq_bands = es.FrequencyBands()

bands_energies = []
for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
    bands_energies.append(freq_bands(s(w(frame))))

novelty = es.NoveltyCurve(
    frameRate=sample_rate / hop_size, weightCurveType=weight
)(numpy.array(bands_energies))
(
    bpm,
    candidates,
    magnitudes,
    tempogram,
    _,
    ticks,
    ticks_strength,
    sinusoid,
) = es.BpmHistogram(frameRate=sample_rate / hop_size)(novelty)

print(f"BPM: {bpm:.1f}")

pylab.plot(novelty)
pylab.suptitle("novelty")
pylab.show()

pylab.matshow(tempogram.T, origin="upper", aspect="auto", interpolation=None)
pylab.suptitle("tempogram")
pylab.show()
