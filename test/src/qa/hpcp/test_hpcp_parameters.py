import numpy as np

# Tuple of probe signals (tones and noise) for testing
testSignals = ('Tones A0 - G9', 
        'Tones A0 - G1', 
        'Tones A1 - G2', 
        'Tones A2 - G3', 
        'Tones A3 - G4', 
        'Tones A4 - G5', 
        'Tones A5 - G6', 
        'Tones A6 - G7', 
        'Tones A7 - G8', 
        'Tones A8 - G9', 
        'White Noise', 
        'LPF WN midi ref. 0', 
        'LPF WN midi ref. 1', 
        'LPF WN midi ref. 2', 
        'LPF WN midi ref. 3', 
        'LPF WN midi ref. 4', 
        'LPF WN midi ref. 5', 
        'LPF WN midi ref. 6', 
        'LPF WN midi ref. 7', 
        'LPF WN midi ref. 8', 
        'LPF WN midi ref. 9', 
        'LPF WN midi ref. 10', 
        'LPF WN midi ref. 11', 
        'BPF WN midi ref. 19', 
        'BPF WN midi ref. 20', 
        'BPF WN midi ref. 21', 
        'BPF WN midi ref. 22', 
        'BPF WN midi ref. 23', 
        'BPF WN midi ref. 24', 
        'BPF WN midi ref. 25', 
        'BPF WN midi ref. 26', 
        'BPF WN midi ref. 27', 
        'BPF WN midi ref. 28', 
        'BPF WN midi ref. 29', 
        'BPF WN midi ref. 30' 
                 )
# Test titles
testTitles = {
    'testWindowWholeRange': 'HPCP and window type in different scales',
    'testFrameSize': 'HPCP, window type and frame size in the lower scales', 
    'testMaxFreq': 'HPCP, window type and max frequency in the upper octaves',
    'testJordiNL': 'Jordi non-linear filtering',
    'testWeigth': 'Weigthing function and window type',
    'testLPFilterMeanSpectrum': 'Mean spectrum of low pass filtered white noise',
    'testBPFilterMeanSpectrum': 'Mean spectrum of band pass filtered white noise',
    'testLPFilterFrameSpeaks': 'Frame spectral peaks of low pass filtered white noise',
    'testBPFilterFrameSpeaks': 'Frame spectral peaks of band pass filtered white noise',
    'testNormalizeWindow': 'Window normalization for different window types',
    'testNormalizeHPCP': 'HPCP normalization',
}

# Mapping between the selected test and the probe 
test2SignalsMapping = {
        'testWindowWholeRange':
            ['Tones A0 - G9', 
            'Tones A0 - G1', 
            'Tones A1 - G2', 
            'Tones A2 - G3', 
            'Tones A3 - G4', 
            'Tones A4 - G5', 
            'Tones A5 - G6', 
            'Tones A6 - G7', 
            'Tones A7 - G8', 
            'Tones A8 - G9'],
        'testFrameSize':
            ['Tones A0 - G1', 
            'Tones A1 - G2', 
            'Tones A2 - G3'],
        'testMaxFreq':
            ['Tones A6 - G7', 
            'Tones A7 - G8', 
            'Tones A8 - G9'],
        'testJordiNL':
            ['Tones A0 - G9'], 
         'testWeigth':
            ['Tones A0 - G9'], 
        'testLPFilterMeanSpectrum':
            ['White Noise',
            'LPF WN midi ref. 0', 
            'LPF WN midi ref. 1', 
            'LPF WN midi ref. 2', 
            'LPF WN midi ref. 3', 
            'LPF WN midi ref. 4', 
            'LPF WN midi ref. 5', 
            'LPF WN midi ref. 6', 
            'LPF WN midi ref. 7', 
            'LPF WN midi ref. 8', 
            'LPF WN midi ref. 9', 
            'LPF WN midi ref. 10', 
            'LPF WN midi ref. 11'],
        'testBPFilterMeanSpectrum':
            ['White Noise',
            'BPF WN midi ref. 19', 
            'BPF WN midi ref. 20', 
            'BPF WN midi ref. 21', 
            'BPF WN midi ref. 22', 
            'BPF WN midi ref. 23', 
            'BPF WN midi ref. 24', 
            'BPF WN midi ref. 25', 
            'BPF WN midi ref. 26', 
            'BPF WN midi ref. 27', 
            'BPF WN midi ref. 28', 
            'BPF WN midi ref. 29', 
            'BPF WN midi ref. 30'],
        'testLPFilterFrameSpeaks':
            ['White Noise',
            'LPF WN midi ref. 0', 
            'LPF WN midi ref. 1', 
            'LPF WN midi ref. 2', 
            'LPF WN midi ref. 3', 
            'LPF WN midi ref. 4', 
            'LPF WN midi ref. 5', 
            'LPF WN midi ref. 6', 
            'LPF WN midi ref. 7', 
            'LPF WN midi ref. 8', 
            'LPF WN midi ref. 9', 
            'LPF WN midi ref. 10', 
            'LPF WN midi ref. 11'],
        'testBPFilterFrameSpeaks':
            ['White Noise',
            'BPF WN midi ref. 19', 
            'BPF WN midi ref. 20', 
            'BPF WN midi ref. 21', 
            'BPF WN midi ref. 22', 
            'BPF WN midi ref. 23', 
            'BPF WN midi ref. 24', 
            'BPF WN midi ref. 25', 
            'BPF WN midi ref. 26', 
            'BPF WN midi ref. 27', 
            'BPF WN midi ref. 28', 
            'BPF WN midi ref. 29', 
            'BPF WN midi ref. 30'],
        'testNormalizeWindow':
            ['Tones A0 - G9'], 
        'testNormalizeHPCP':
            ['Tones A0 - G9'], 
                           }

# General parameters for signals creation
fs = 44100.
t = 3 * 60.  # 3 minutes
N = t * fs
f0 = 440.
sampling = np.arange(int(N))
semitone = np.power(2.,1./12.)
f0 = 440.  # A4 => 440 Hz
precision = 0.0001

# Mapping between tests and  the tones of the signal
# The tones are expressed as de distance in semitones to the central A 440
note_scales = {
        'Tones A0 - G9': [- 5 * 12 , -1 + 4 * 12], 
        'Tones A0 - G1': [- 5 * 12 , -1 - 4 * 12], 
        'Tones A1 - G2': [- 4 * 12 , -1 - 3 * 12], 
        'Tones A2 - G3': [- 3 * 12 , -1 - 2 * 12], 
        'Tones A3 - G4': [- 2 * 12 , -1 - 1 * 12], 
        'Tones A4 - G5': [- 1 * 12 , -1 - 0 * 12], 
        'Tones A5 - G6': [- 0 * 12 , -1 + 1 * 12], 
        'Tones A6 - G7': [+ 1 * 12 , -1 + 2 * 12], 
        'Tones A7 - G8': [+ 2 * 12 , -1 + 3 * 12],
        'Tones A8 - G9': [+ 3 * 12 , -1 + 4 * 12]
}


# Default common parameters for testing
hpcpSize = 3 * 12
win_normalized = False
frameSize_ = 4096
hopSize_ = frameSize_ / 2    
speaks_max = 60
maxFrequencyList = 12000
nonLinear = False
weightTypes = 'none'
hpcp_normalized = 'none'
