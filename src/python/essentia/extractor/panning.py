import essentia
from essentia import INFO
import sys
import math
from math import *


namespace = 'panning'
dependencies = None


def compute(audio, pool, options):

    INFO('Computing Panning descriptors...')

    filename = pool.descriptors['metadata']['filename']['values'][0]
    sampleRate = options['sampleRate']
    frameSize = options['frameSize']
    hopSize = options['hopSize']

    audioLeft, audioRight, originalSampleRate, originalChannelsNumber = essentia.AudioFileInput(filename = filename,
                                                                                                outputSampleRate = sampleRate,
                                                                                                stereo = 'True')()
    # in case of a mono file
    if originalChannelsNumber == 1:
        audioRight = audioLeft

    panning = essentia.ExtractorPanning(frameSize = frameSize, hopSize = hopSize)
    coefficients = panning(audioLeft, audioRight);

    # used for a nice progress display
    total_frames = len(coefficients)
    n_frames = 0
    start_of_frame = -frameSize*0.5

    progress = essentia.Progress(total = total_frames)

    while n_frames < total_frames:

        frameScope = [ start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate ]
        pool.setCurrentScope(frameScope)

        pool.add('coefficients', essentia.array(coefficients[n_frames]), frameScope)

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    progress.finish()

