import sys
from essentia.standard import *
import pylab as plt

try:
    input_file = sys.argv[1]
except:
    print(f"usage: {sys.argv[0]} <input_file>")
    sys.exit()

audio = MonoLoader(filename=input_file)()
bpm, _, _, _, intervals = RhythmExtractor2013()(audio)

(
    peak1_bpm,
    peak1_weight,
    peak1_spread,
    peak2_bpm,
    peak2_weight,
    peak2_spread,
    histogram,
) = BpmHistogramDescriptors()(intervals)

print(f"Overall BPM: {bpm:.1f}")
print(f"First peak: {peak1_bpm:.1f} bpm")
print(f"Second peak: {peak2_bpm:.1f} bpm")

fig, ax = plt.subplots()
ax.bar(range(len(histogram)), histogram, width=1)
ax.set_xlabel("BPM")
ax.set_ylabel("Frequency")
ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
plt.show()
