import essentia
from essentia import INFO
from numpy import bincount


namespace = 'rhythm'
dependencies = [ 'tempotap', 'onsetdetection' ]


def compute(audio, pool, options):

    INFO('Computing Inter Onsets Intervals...')

    sampleRate = options['sampleRate']
    bpm = pool.value('rhythm.bpm')
    onsets = pool.value('rhythm.onset_times')

    # special case
    if bpm < 0 or len(onsets) < 2:
       pool.add(namespace + '.' + 'relative_ioi_peaks', [float()])#, pool.GlobalScope)
       pool.add(namespace + '.' + 'relative_ioi', [float()])#, pool.GlobalScope)

       INFO('100% done...')

       return

    # 32th note interval
    interp = 32.
    interval = (60./bpm) / interp
    riois = []
    old = onsets[0]
    for i in range(1,len(onsets)): riois += [ round( (onsets[i] - onsets[i-1]) / interval ) ]
    for i in range(2,len(onsets)): riois += [ round( (onsets[i] - onsets[i-2]) / interval ) ]
    for i in range(3,len(onsets)): riois += [ round( (onsets[i] - onsets[i-3]) / interval ) ]
    for i in range(4,len(onsets)): riois += [ round( (onsets[i] - onsets[i-4]) / interval ) ]
    ioidist = essentia.array(bincount(riois))
    fullioidist = essentia.array(zip( [p/interp for p in range(len(ioidist))], [ioi/sum(ioidist) for ioi in ioidist]))
    fullioidist = fullioidist[0:interp*5]
    peak_detection = essentia.PeakDetection(minPosition = 0., maxPosition = len(ioidist),
                                            maxPeaks = 5, range = len(ioidist) - 1.,
                                            interpolate = True, orderBy = 'amplitude')
    pos, mags = peak_detection(ioidist)

    # scale back to 1 beat
    pos = [ p/interp for p in pos ]

    # ratio across whole distribution surface
    mags = [ mag/sum(ioidist) for mag in mags ]

    # add to pool
    pool.add(namespace + '.' + 'relative_ioi_peaks', essentia.array(zip(pos,mags)))#, pool.GlobalScope)
    pool.add(namespace + '.' + 'relative_ioi', fullioidist)#, pool.GlobalScope)

    # debug plot
    if 0:
        from pylab import plot, show, hold
        plot([i/interp for i in range(len(ioidist))], [ioi/sum(ioidist) for ioi in ioidist],'b+-')
        hold(True)
        for i,j in zip(pos,mags):
            plot([i]*2,[0.,j],'+-')
        hold(False)
        show()

    INFO('100% done...')
