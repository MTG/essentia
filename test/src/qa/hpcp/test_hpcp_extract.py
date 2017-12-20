import numpy as np
from scipy.signal import spectrogram

import essentia
from essentia.standard import *

from test_hpcp_parameters import *


def extractHPCP(audiosignal, frameSize, hopSize, w, speaks, hpcp, signalname):
    # w is the preconfigured windowing algorithm
    # hpcp is the preconfigured HPCP algorithm 

    audio = essentia.array(audiosignal)
    # TODO: not sure if this is necessary: 
    if len(audio)%2:
        audio = audio[:-1] 

    spectrum = Spectrum()
    speaks.maxFrequency = hpcp.paramValue('maxFrequency')
    chromagram = []
    spectrogram = []

    signal_spectrum = spectrum(audio)

    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame_spectrum = spectrum(w(frame))
        spectrogram.append(frame_spectrum)
        pfreq, pmagn = speaks(frame_spectrum)
        chromagram.append(hpcp(pfreq, pmagn))

    spectrogram = essentia.array(spectrogram).T    
    chromagram = essentia.array(chromagram).T
       
    hpcp_mean = np.mean(chromagram, axis=1)
    hpcp_median = np.median(chromagram, axis=1)
    
    
    return chromagram, spectrogram, signal_spectrum, hpcp_mean, hpcp_median


