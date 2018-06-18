import sys
from essentia.standard import *
from essentia import Pool
import numpy

try:
    input_file = sys.argv[1]
except:
    print("Estimates BPM using RhythmTransform")
    print("usage: %s <input_file>" % sys.argv[0])
    sys.exit()


"""
Explanation of Rhythm Transform: 
- Mel bands are computed on frames of the size 8192 with the frames sample rate = sampleRate/hopSize = 22050/1024 = 21.5Hz
- Rhythm transform frame size is equal to 256 Mel bands frames
- Output vector is of size 256/2 + 1 = 129.
- Therefore it represents periodicities over the interval 0Hz (0th bin) to 22050/1024/2 = 10.75Hz (129th bin),
- Converting to BPM values this corresponds to an interval from 0 BPM to 22050/1024/2 * 60 = 646 BPM
- Each bin roughly covers 5 BPM
- 60-200 BPM interval is covered by only 40-12 = 28 bins
- 120 BPM rougphly corresponds to bin #24
- bin 0 = 0 BPM
- bin 128 = 645.99609375 BPM
"""

sampleRate   = 22050
frameSize    = 8192
hopSize      = 1024
rmsFrameSize = 256
rmsHopSize   = 32

loader = MonoLoader(filename=input_file, sampleRate=sampleRate)
w = Windowing(type='blackmanharris62')
spectrum = Spectrum()
melbands = MelBands(sampleRate=sampleRate, numberBands=40, lowFrequencyBound=0, highFrequencyBound=sampleRate/2)

pool = Pool()

for frame in FrameGenerator(audio=loader(), frameSize=frameSize, hopSize=hopSize, startFromZero=True):
    bands = melbands(spectrum(w(frame)))
    pool.add('melbands', bands)

rhythmtransform = RhythmTransform(frameSize=rmsFrameSize, hopSize=rmsHopSize)
rt = rhythmtransform(pool['melbands'])
rt_mean = numpy.mean(rt, axis=0)
bin_resoluion = 5.007721656976744

print("Estimated BPM: %0.1f" % float(numpy.argmax(rt_mean) * bin_resoluion))
