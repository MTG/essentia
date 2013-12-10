# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

import sys, csv
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *

try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<input-audiofile>"
    sys.exit()



# In this example we will extract predominant melody given an audio file by
# running a chain of algorithms.
 
# First, create our algorithms:
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True

run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
run_spectrum = Spectrum(size=frameSize * 4)
run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                   maxFrequency=20000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")
run_pitch_salience_function = PitchSalienceFunction()
run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
run_pitch_contours = PitchContours(hopSize=hopSize)
run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                hopSize=hopSize)

# ... and create a Pool
pool = Pool();

# Now we are ready to start processing.
# 1. Load audio and pass it through the equal-loudness filter
audio = MonoLoader(filename = filename)()
audio = EqualLoudness()(audio)

# 2. Cut audio into frames and compute for each frame:
#    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    frame = run_windowing(frame)
    spectrum = run_spectrum(frame)
    peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
    
    salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
    salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
    
    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

# 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
#    tracking and melody detection algorithms:
contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
        pool['allframes_salience_peaks_bins'],
        pool['allframes_salience_peaks_saliences'])
pitch, confidence = run_pitch_contours_melody(contours_bins,
                                              contours_saliences,
                                              contours_start_times,
                                              duration)

# NOTE that we can avoid the majority of intermediate steps by using a composite algorithm
#      PredominantMelody (see extractor_predominant_melody.py). This script will be usefull 
#      if you want to get access to pitch salience function and pitch contours.

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
