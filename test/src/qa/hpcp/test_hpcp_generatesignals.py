import random
import numpy as np
from collections import defaultdict

from scipy.signal import spectrogram
import essentia
from essentia.standard import *

from test_hpcp_parameters import *


def getSignalsList(tests):
    """
    getSignalsList(['testWindowWholeRange', 'testFrameSize']])
    """
    signals = {}
    signalsKey = []
    
    for test in tests:
        for signalNeeded in test2SignalsMapping[test]:
            if not signalNeeded in signalsKey:
                signals[signalNeeded]=getSignal(signalNeeded)
                signalsKey.append(signalNeeded)
    
    return signals, signalsKey
    
def getSignal(signalNeeded):
    if signalNeeded.startswith('Tones'):
        return (getTone(signalNeeded))
    elif signalNeeded.startswith('White Noise'):
        return (getWhiteNoise())
    elif signalNeeded.startswith('LPF'):
        return (getLPF(signalNeeded))
    elif signalNeeded.startswith('BPF'):
        return (getBPF(signalNeeded))
    else:
        print ('Error in getSignal')
        return ([])
    

def getTone(signalNeeded):

    # Note scales
    N_offset0 = float(note_scales[signalNeeded][0])
    N_offset1 = float(note_scales[signalNeeded][1])
    nnotes = N_offset1 - N_offset0 + 1

    samples_per_note = np.rint((float(N) / float(nnotes)) / 2.0)
	# Divide in half because we are adding up silences between notes

    scale_midi = np.arange(N_offset0, N_offset1+1)
    scale_frequencies = 440.*np.power(semitone, scale_midi)

    number_of_samples_for_period = np.rint (fs / scale_frequencies)
    number_of_samples_for_note = (np.rint(samples_per_note/number_of_samples_for_period)+1)*number_of_samples_for_period-1
    number_of_samples_of_silence = 2 * samples_per_note - number_of_samples_for_note

    # Build the tones and the silences
    # First in signal we distribute the phases of the different tones
    signal = []
    for i in xrange(len(scale_frequencies)):
        y = np.repeat(scale_frequencies[i],number_of_samples_for_note[i])
        signal = np.insert(signal,len(signal),y,axis=0)
        y = np.repeat(0.,number_of_samples_of_silence[i])
        signal = np.insert(signal,len(signal),y,axis=0)

    # Now we calculate the sine of the phases
    signal = np.sin(2*np.pi*np.arange(len(signal))/fs*signal)
    # Normalization
    signal = signal / max(signal)

    return signal
                
def getWhiteNoise():
    signal = essentia.array(np.random.randn(int(N)))
    # Normalization
    signal = signal / max(signal)
    
    return signal


def getLPF(signalNeeded):
    
    midi = int(float(signalNeeded[-2:]))
    
    fc = 440.*np.power(semitone, midi)
    LPF = LowPass(sampleRate=fs, cutoffFrequency=fc)
    signal = essentia.array(LPF(getWhiteNoise()))
    # Normalization
    signal = signal / max(signal)
    
    return signal

def getBPF(signalNeeded):
    
    midi = int(float(signalNeeded[-2:]))
    
    
    fc0 = 440.*np.power(semitone, midi)
    fc1 = 440.*np.power(semitone, midi + 2*12 + 1) # 2 octaves
    bandwidth = fc1 - fc0
    cutoffFrequency = (fc0 + fc1) / 2.
    BPF = BandPass(bandwidth=bandwidth, cutoffFrequency=cutoffFrequency,sampleRate=fs)
    signal = essentia.array(BPF(getWhiteNoise()))    
    # Normalization
    signal = signal / max(signal)
    
    return signal
