import essentia
import numpy
import sys
from essentia import INFO
from essentia.progress import Progress

namespace = 'rhythm'
dependencies = None


def compute(audio, pool, options):

    sampleRate  = options['sampleRate']
    frameSize   = options['frameSize']
    hopSize     = options['hopSize']
    zeroPadding = options['zeroPadding']
    windowType  = options['windowType']

    frameRate = float(sampleRate)/float(frameSize - hopSize)

    INFO('Computing Onset Detection...')

    frames  = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window  = essentia.Windowing(size = frameSize, zeroPadding = zeroPadding, type = windowType)
    fft = essentia.FFT()
    cartesian2polar = essentia.CartesianToPolar()
    onsetdetectionHFC = essentia.OnsetDetection(method = "hfc", sampleRate = sampleRate)
    onsetdetectionComplex = essentia.OnsetDetection(method = "complex", sampleRate = sampleRate)
    onsets = essentia.Onsets(frameRate = frameRate)

    total_frames = frames.num_frames()
    n_frames = 0
    start_of_frame = -frameSize*0.5

    hfc = []
    complex = []

    progress = Progress(total = total_frames)

    for frame in frames:

        if essentia.instantPower(frame) < 1.e-4 :
           total_frames -= 1
           start_of_frame += hopSize
           hfc.append(0.)
           complex.append(0.)
           continue

        windowed_frame = window(frame)
        complex_fft = fft(windowed_frame)
        (spectrum,phase) = cartesian2polar(complex_fft)
        hfc.append(onsetdetectionHFC(spectrum,phase))
        complex.append(onsetdetectionComplex(spectrum,phase))

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    # The onset rate is defined as the number of onsets per seconds
    detections = numpy.concatenate([essentia.array([hfc]), essentia.array([complex]) ])

    # prune all 'doubled' detections
    time_onsets = list(onsets(detections, essentia.array([1, 1])))
    t = 1
    while t < len(time_onsets):
      if time_onsets[t] - time_onsets[t-1] < 0.080: time_onsets.pop(t)
      else: t += 1

    onsetrate = len(time_onsets) / ( len(audio) / sampleRate )

    pool.add(namespace + '.' + "onset_times", essentia.array(time_onsets))#, pool.GlobalScope)
    pool.add(namespace + '.' + "onset_rate", onsetrate)#, pool.GlobalScope)

    progress.finish()
