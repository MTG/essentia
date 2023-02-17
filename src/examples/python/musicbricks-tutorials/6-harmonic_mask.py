# import essentia in standard mode
import essentia.standard as es

# We'll need to some numerical tools and to define filepaths
import numpy as np
from pathlib import Path

# input and output files
tutorial_dir = Path(__file__).resolve().parent
inputFilename = tutorial_dir / "flamenco.wav"
outputFilename = tutorial_dir / "flamenco_mask.wav"

# algorithm parameters
framesize = 2048
hopsize = 128  # PredominantPitchMelodia requires a hopsize of 128
samplerate = 44100.0
attenuation_dB = 100
maskbinwidth = 2

# create an audio loader and import audio file
audio = es.MonoLoader(filename=str(inputFilename), sampleRate=samplerate)()
print(f"Duration of the audio sample [sec]: {len(audio) / samplerate:.3f}")


# extract predominant pitch
# PitchMelodia takes the entire audio signal as input - no frame-wise processing is required here.
pExt = es.PredominantPitchMelodia(
    frameSize=framesize, hopSize=hopsize, sampleRate=samplerate
)
pitch, pitchConf = pExt(audio)


# algorithm workflow for harmonic mask using the STFT frame-by-frame
fcut = es.FrameCutter(frameSize=framesize, hopSize=hopsize)
w = es.Windowing(type="hann")
fft = es.FFT(size=framesize)
hmask = es.HarmonicMask(
    sampleRate=samplerate, binWidth=maskbinwidth, attenuation=attenuation_dB
)
ifft = es.IFFT(size=framesize)
overl = es.OverlapAdd(frameSize=framesize, hopSize=hopsize)
awrite = es.MonoWriter(filename=str(outputFilename), sampleRate=samplerate)


# init output audio array
audioout = np.array(0)

# loop over all frames
for idx, frame in enumerate(
    es.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
):

    # STFT analysis
    infft = fft(w(frame))
    # get pitch of current frame
    curpitch = pitch[idx]

    # here we  apply the harmonic mask spectral transformations
    outfft = hmask(infft, pitch[idx])

    # STFT synthesis
    out = overl(ifft(outfft))
    audioout = np.append(audioout, out)


# write audio output
awrite(audioout.astype(np.float32))
