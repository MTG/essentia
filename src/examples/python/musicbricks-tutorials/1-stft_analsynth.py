import essentia
import essentia.streaming as es
from pathlib import Path

# algorithm parameters
framesize = 1024
hopsize = 256

# input and output files
tutorial_dir = Path(__file__).resolve().parent
inputFilename = tutorial_dir / "singing-female.wav"
outputFilename = tutorial_dir / "singing-female-out-stft.wav"

# initialize algorithms
loader = es.MonoLoader(filename=str(inputFilename), sampleRate=44100)
fcut = es.FrameCutter(frameSize=framesize, hopSize=hopsize, startFromZero=False)
w = es.Windowing(type="hann")
fft = es.FFT(size=framesize)
ifft = es.IFFT(size=framesize)
overl = es.OverlapAdd(
    frameSize=framesize, hopSize=hopsize, gain=1.0 / framesize
)
awrite = es.MonoWriter(filename=str(outputFilename), sampleRate=44100)
pool = essentia.Pool()

# define a network of connected algorithms using outputs and inputs
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> ifft.fft
ifft.frame >> overl.frame
overl.signal >> awrite.audio
overl.signal >> (pool, "audio")

# starting the network
essentia.run(loader)
