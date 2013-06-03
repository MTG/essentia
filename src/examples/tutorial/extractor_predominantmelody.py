import sys, csv
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
from math import log
from math import floor

try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<input-audiofile>"
    sys.exit()

hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True

# RUNNING A COMPOSITE ALGORITHM 
run_predominant_melody = PredominantMelody(guessUnvoiced=guessUnvoiced, 
                                           frameSize=frameSize,
                                           hopSize=hopSize);

# load audio file, apply equal loudness filter, and compute predominant melody
audio = MonoLoader(filename = filename, sampleRate=sampleRate)()
audio = EqualLoudness()(audio)
pitch, confidence = run_predominant_melody(audio)


n_frames = len(pitch)
print "number of frames:", n_frames

# visualize output pitch
fig = plt.figure()
plot(range(n_frames), pitch, 'b')
n_ticks = 10
xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch (Hz)')
suptitle("Predominant melody pitch")

# visualize output pitch confidence
fig = plt.figure()
plot(range(n_frames), confidence, 'b')
n_ticks = 10
xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Confidence')
suptitle("Predominant melody pitch confidence")

show()
