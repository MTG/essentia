import essentia
import numpy
import sys
from essentia import INFO
from essentia.progress import Progress

namespace = 'lowlevel'
dependencies = None


def is_silent_threshold(frame, silence_threshold_dB):
    p = essentia.instantPower( frame )
    silence_threshold = pow(10.0, (silence_threshold_dB / 10.0))
    if p < silence_threshold:
       return 1.0
    else:
       return 0.0


def compute(audio, pool, options):

    # analysis parameters
    sampleRate = options['sampleRate']
    frameSize  = options['frameSize']
    hopSize    = options['hopSize']
    windowType = options['windowType']

    # frame algorithms
    frames = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window = essentia.Windowing(size = frameSize, zeroPadding = 0, type = windowType)
    spectrum = essentia.Spectrum(size = frameSize)

    # spectral algorithms
    energy = essentia.Energy()
    mfcc = essentia.MFCC(highFrequencyBound = 8000)

    INFO('Computing Low-Level descriptors necessary for segmentation...')

    # used for a nice progress display
    total_frames = frames.num_frames()
    n_frames = 0
    start_of_frame = -frameSize*0.5

    progress = Progress(total = total_frames)

    for frame in frames:

        frameScope = [ start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate ]
        #pool.setCurrentScope(frameScope)
        pool.add(namespace + '.' + 'scope', frameScope)

        if options['skipSilence'] and essentia.isSilent(frame):
          total_frames -= 1
          start_of_frame += hopSize
          continue

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        # need the energy for getting the thumbnail
        pool.add(namespace + '.' + 'spectral_energy', energy(frame_spectrum))

        # mfcc
        (frame_melbands, frame_mfcc) = mfcc(frame_spectrum)
        pool.add(namespace + '.' + 'spectral_mfcc', frame_mfcc)

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    progress.finish()
