import sys

from essentia.streaming import *


try:
	audiofile = sys.argv[1]
except:
	print "usage:", sys.argv[0], "<audiofile>"
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
print "BPM:", pool['bpm']
print "BPM histogram:"
print pool['bpm_histogram']
print "Most prominent peak:", pool['bpm_first_peak'], "BPM"
print "Centroid:", pool['bpm_centroid']