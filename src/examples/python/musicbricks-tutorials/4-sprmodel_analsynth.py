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


# define input and output files
tutorial_dir = Path(__file__).resolve().parent
inputFilename = tutorial_dir / "singing-female.wav"
outputFilename = tutorial_dir / "singing-female-out-sprmodel.wav"


# initialize some algorithms to define a network of algorithms
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
smanal = es.SprModelAnal(
    sampleRate=params["sampleRate"],
    maxnSines=params["maxnSines"],
    magnitudeThreshold=params["magnitudeThreshold"],
    freqDevOffset=params["freqDevOffset"],
    freqDevSlope=params["freqDevSlope"],
)
synFFTSize = int(min(int(params["frameSize"] / 4), 4 * params["hopSize"]))

# make sure the FFT size is appropriate
smsyn = es.SprModelSynth(
    sampleRate=params["sampleRate"],
    fftSize=synFFTSize,
    hopSize=params["hopSize"],
)
# We'll need a pool to store the results
pool = essentia.Pool()


# analysis
loader.audio >> fcut.signal
fcut.frame >> smanal.frame
smanal.magnitudes >> (pool, "magnitudes")
smanal.frequencies >> (pool, "frequencies")
smanal.phases >> (pool, "phases")

# synthesis
smanal.magnitudes >> smsyn.magnitudes
smanal.frequencies >> smsyn.frequencies
smanal.phases >> smsyn.phases
smanal.res >> smsyn.res

smsyn.frame >> (pool, "frames")
smsyn.sineframe >> (pool, "sineframes")
smsyn.resframe >> (pool, "resframes")


essentia.run(loader)


# store to file
outaudio = pool["frames"].flatten()

awrite = es.MonoWriter(
    filename=str(outputFilename), sampleRate=params["sampleRate"]
)
outvector = es.VectorInput(outaudio)

outvector.data >> awrite.audio
essentia.run(outvector)
