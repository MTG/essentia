import sys
from essentia.standard import *
from essentia import Pool

try:
    input_file = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<input_file>"
    sys.exit()

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



print len(pool['melbands']), "Mel band frames"
print len(pool['melbands']) / 32, "Rhythm transform frames"


rhythmtransform = RhythmTransform(frameSize=rmsFrameSize, hopSize=rmsHopSize)
rt = rhythmtransform(pool['melbands'])

import matplotlib.pyplot as plt
plt.imshow(rt.T[:,:], aspect = 'auto')
plt.xlabel('Frames')
plt.ylabel('Rhythm Transform coefficients')
plt.show()


