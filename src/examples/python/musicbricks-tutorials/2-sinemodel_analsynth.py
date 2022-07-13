import essentia
import essentia.streaming as es
from pathlib import Path

# algorithm parameters
params = {
    "frameSize": 2048,
    "hopSize": 512,
    "startFromZero": False,
    "sampleRate": 44100,
    "maxnSines": 100,
    "magnitudeThreshold": -74,
    "minSineDur": 0.02,
    "freqDevOffset": 10,
    "freqDevSlope": 0.001,
}

# input and output files
tutorial_dir = Path(__file__).resolve().parent
inputFilename = tutorial_dir / "singing-female.wav"
outputFilename = tutorial_dir / "singing-female-out-sinemodel.wav"

# initialize some algorithms
loader = es.MonoLoader(
    filename=str(inputFilename), sampleRate=params["sampleRate"]
)
fcut = es.FrameCutter(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    startFromZero=False,
)
w = es.Windowing(type="blackmanharris92")
fft = es.FFT(size=params["frameSize"])
smanal = es.SineModelAnal(
    sampleRate=params["sampleRate"],
    maxnSines=params["maxnSines"],
    magnitudeThreshold=params["magnitudeThreshold"],
    freqDevOffset=params["freqDevOffset"],
    freqDevSlope=params["freqDevSlope"],
)
smsyn = es.SineModelSynth(
    sampleRate=params["sampleRate"],
    fftSize=params["frameSize"],
    hopSize=params["hopSize"],
)
ifft = es.IFFT(size=params["frameSize"])
overl = es.OverlapAdd(frameSize=params["frameSize"], hopSize=params["hopSize"])
awrite = es.MonoWriter(
    filename=str(outputFilename), sampleRate=params["sampleRate"]
)
pool = essentia.Pool()


# Define a network of connected algorithms

# analysis
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> smanal.fft
smanal.magnitudes >> (pool, "magnitudes")
smanal.frequencies >> (pool, "frequencies")
smanal.phases >> (pool, "phases")

# synthesis
smanal.magnitudes >> smsyn.magnitudes
smanal.frequencies >> smsyn.frequencies
smanal.phases >> smsyn.phases
smsyn.fft >> ifft.fft
ifft.frame >> overl.frame
overl.signal >> awrite.audio
overl.signal >> (pool, "audio")

# run the network
essentia.run(loader)
