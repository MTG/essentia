import essentia
import essentia.streaming as es
from pathlib import Path


# algorithm parameters
params = {
    "frameSize": 2048,
    "hopSize": 128,
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
outputFilename = tutorial_dir / "singing-female-out-sinesubtraction.wav"

# initialize some algorithms to define a network of algorithms
loader = es.MonoLoader(
    filename=str(inputFilename), sampleRate=params["sampleRate"]
)
pool = essentia.Pool()
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
subtrFFTSize = int(min(params["frameSize"] / 4, 4 * params["hopSize"]))
smsub = es.SineSubtraction(
    sampleRate=params["sampleRate"],
    fftSize=subtrFFTSize,
    hopSize=params["hopSize"],
)

# Define a network of connected algorithms

# analysis
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> smanal.fft
smanal.magnitudes >> (pool, "magnitudes")
smanal.frequencies >> (pool, "frequencies")
smanal.phases >> (pool, "phases")
# subtraction
fcut.frame >> smsub.frame
smanal.magnitudes >> smsub.magnitudes
smanal.frequencies >> smsub.frequencies
smanal.phases >> smsub.phases
smsub.frame >> (pool, "frames")

# run network
essentia.run(loader)


# store the results in a file with a new network of algorithms
outaudio = pool["frames"].flatten()

# initialize some algorithms
awrite = es.MonoWriter(
    filename=str(outputFilename), sampleRate=params["sampleRate"]
)
outvector = es.VectorInput(outaudio)

# define new network
outvector.data >> awrite.audio

# run network to write audio file
essentia.run(outvector)
