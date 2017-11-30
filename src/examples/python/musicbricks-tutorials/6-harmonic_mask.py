# import essentia in standard mode
import essentia
import essentia.standard
from essentia.standard import *

# import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

# input and output files
import os.path
tutorial_dir = os.path.dirname(os.path.realpath(__file__))
inputFilename = os.path.join(tutorial_dir, 'flamenco.wav')
outputFilename = os.path.join(tutorial_dir, 'flamenco_mask.wav')

# algorithm parameters
framesize = 2048
hopsize = 128 #  PredominantPitchMelodia requires a hopsize of 128
samplerate = 44100.0
attenuation_dB = 100
maskbinwidth = 2

# create an audio loader and import audio file
loader = essentia.standard.MonoLoader(filename=inputFilename, sampleRate=samplerate )
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/ samplerate )


#extract predominant pitch
# PitchMelodia takes the entire audio signal as input - no frame-wise processing is required here.
pExt = PredominantPitchMelodia(frameSize=framesize, hopSize=hopsize, sampleRate=samplerate)
pitch, pitchConf = pExt(audio)


# algorithm workflow for harmonic mask using the STFT frame-by-frame
fcut = FrameCutter(frameSize=framesize, hopSize=hopsize);
w = Windowing(type="hann");
fft = FFT(size=framesize);
hmask = HarmonicMask( sampleRate=samplerate, binWidth=maskbinwidth, attenuation=attenuation_dB);
ifft = IFFT(size=framesize);
overl = OverlapAdd(frameSize=framesize, hopSize=hopsize);
awrite = MonoWriter (filename=outputFilename, sampleRate=44100);


# init output audio array
audioout = np.array(0)

# loop over all frames
for idx, frame in enumerate(FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)):

    # STFT analysis
    infft = fft(w(frame))
    # get pitch of current frame
    curpitch = pitch[idx]

    # here we  apply the harmonic mask spectral transformations
    outfft = hmask(infft, pitch[idx]);

    # STFT synthesis
    out = overl(ifft(outfft))
    audioout = np.append(audioout, out)


# write audio output
awrite(audioout.astype(np.float32))



