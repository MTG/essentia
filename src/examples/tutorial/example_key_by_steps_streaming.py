import sys
import essentia
from essentia.streaming import *
from essentia.standard import YamlOutput

try:
    infile = sys.argv[1]
    outfile = sys.argv[2]
except:
    print "usage:", sys.argv[0], "<input audio file> <output json file>"
    sys.exit()

# initialize algorithms we will use
loader = MonoLoader(filename=infile)
framecutter = FrameCutter()
windowing = Windowing(type="blackmanharris62")
spectrum = Spectrum()
spectralpeaks = SpectralPeaks(orderBy="magnitude",
                              magnitudeThreshold=1e-05,
                              minFrequency=40,
                              maxFrequency=5000, 
                              maxPeaks=10000)
hpcp = HPCP()
key = Key()

# use pool to store data
pool = essentia.Pool() 

# connect algorithms together
loader.audio >> framecutter.signal
framecutter.frame >> windowing.frame >> spectrum.frame
spectrum.spectrum >> spectralpeaks.spectrum
spectralpeaks.magnitudes >> hpcp.magnitudes
spectralpeaks.frequencies >> hpcp.frequencies
hpcp.hpcp >> key.pcp
key.key >> (pool, 'tonal.key_key')
key.scale >> (pool, 'tonal.key_scale')
key.strength >> (pool, 'tonal.key_strength')

# network is ready, run it
essentia.run(loader)

print pool['tonal.key_key'] + " " + pool['tonal.key_scale']

# write to json file
YamlOutput(filename=outfile, format="json")(pool)

