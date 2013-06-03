#!/usr/bin/env python

from essentia import *
from essentia.standard import *

def extractor(filename):
    # load our audio into an array
    audio = MonoLoader(filename = filename)()

    # create the pool and the necessary algorithms
    pool = Pool()
    w = Windowing()
    spec = Spectrum()
    centroid = SpectralCentroid()

    # compute the centroid for all frames in our audio and add it to the pool
    for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
        c = centroid(spec(w(frame)))
        pool.add('lowlevel.centroid', c)


    # aggregate the results
    aggrpool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

    # write result to file
    YamlOutput(filename = filename + '.features.yaml')(aggrpool)


# some python magic so that this file works as a script as well as a module
# if you do not understand what that means, you don't need to care
if __name__ == '__main__':
    import sys
    print 'Script %s called with arguments: %s' % (sys.argv[0], sys.argv[1:])

    try:
        extractor(sys.argv[1])
        print 'Success!'

    except KeyError:
        print 'ERROR: You need to call this script with a filename argument...'
