import sys
import pylab as plt
from essentia.streaming import *


try:
	audiofile = sys.argv[1]
except:
	print ("usage: %s <audiofile>" % sys.argv[0])
	sys.exit()

pool = essentia.Pool()

loader = MonoLoader(filename = audiofile)
bt = RhythmExtractor2013()
bpm_histogram = BpmHistogramDescriptors()
centroid = Centroid(range=250) # BPM histogram output size is 250

loader.audio >> bt.signal
bt.bpm >> (pool, 'bpm')
bt.ticks >> None
bt.confidence >> None
bt.estimates >> None
bt.bpmIntervals >> bpm_histogram.bpmIntervals
bpm_histogram.firstPeakBPM >> (pool, 'bpm_first_peak')
bpm_histogram.firstPeakWeight >> None
bpm_histogram.firstPeakSpread >> None
bpm_histogram.secondPeakBPM >> (pool, 'bpm_second_peak')
bpm_histogram.secondPeakWeight >> None
bpm_histogram.secondPeakSpread >> None
bpm_histogram.histogram >> (pool, 'bpm_histogram')
bpm_histogram.histogram >> centroid.array
centroid.centroid >> (pool, 'bpm_centroid')

essentia.run(loader)
print("BPM: %0.1f" % pool['bpm'])
print("Most prominent peak: %0.1f BPM" % pool['bpm_first_peak'][0])
print("Centroid: %0.1f" % pool['bpm_centroid'][0]) 

histogram = pool['bpm_histogram'][0]

fig, ax = plt.subplots()
ax.bar(range(len(histogram)), histogram, width=1)
ax.set_xlabel('BPM')
ax.set_ylabel('Frequency')
ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
plt.show()