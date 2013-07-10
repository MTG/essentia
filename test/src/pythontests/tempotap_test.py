# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

#! /usr/bin/python

# example script to compute and plot tempo related descriptors

def plot_frame(features_array, ground_truth, sampleRate, hop_size, acf, periods, phases, bin2hz,minlag,maxlag, mcomb):
    import numpy
    from pylab import figure, subplot, imshow, plot, axis, hold, clf, show

    # norm features, acf and peaks
    pfeatures = numpy.array(features_array).transpose()
    maxis = pfeatures.max(axis=1)
    for i in range(len(maxis)): pfeatures[i] /= maxis[i]
    normed_acf = []
    for i in range(len(acf)):
        normed_acf.append(acf[i]/max(acf[i]))
    acf = normed_acf

    # begin plot
    figure()
    clf()

    subplot(311)
    hold(True)
    for i in range(len(pfeatures)): plot(pfeatures[i] + i)
    hold(False)
    axis('tight')

    subplot(312)
    hold(True)
    for i in range(len(mcomb)):
        plot(mcomb[i]/mcomb[i].max() + i)
        plot([mcomb[i].argmax()] * 2, [i, i+1] )
        plot([periods[i]] * 2, [i, i+1] )
    # plot the ground truth
    if ground_truth != None:
        plot([bpmtolag(ground_truth,sampleRate,hop_size)]*2,[0.,len(acf)],'r-')
    # plot the bpm estimate
    for i in range(len(periods)): plot([periods[i]]*2,[i,i+1],'g-')
    hold(False)
    axis('tight')

    subplot(313)
    hold(True)
    for i in range(len(pfeatures)):
      periodnum = 4
      phout = [ 0. for j in range( len(pfeatures[i]) / periodnum) ]
      if periods[i] == 0: continue
      for j in range( len( phout ) ):
        for a in range ( periodnum ):
          phout[j] += pfeatures[i][a*periods[i] + j]
      plot (phout/max(phout) + i)
      phase = phases[i]
      if phase >= periods[i]:
        while phase >= periods[i]:
          phase -= periods[i]
      while phase < len(pfeatures[i]):
          plot([phase]*2,[i,i+1], 'r-')
          phase += periods[i]
    hold(False)
    axis([0., len(pfeatures[i]), 0., len(pfeatures)])

    show()

def plot_bpm_file(pool):
    bpm    = pool.descriptors['tempotap_bpm']['values'][0]
    intervals = pool.descriptors['tempotap_intervals']['values'][0]
    bpm_periods = [60./interval for interval in intervals]
    ticks  = pool.descriptors['tempotap_ticks']['values'][0]
    rubato_start = pool.descriptors['tempotap_rubato_start']['values'][0]
    rubato_stop = pool.descriptors['tempotap_rubato_stop']['values'][0]
    print 'bpm', bpm
    print 'ticks', ticks
    print 'rubato_start', rubato_start
    print 'rubato_stop', rubato_stop
    print 'intervals', intervals
    import pylab
    pylab.plot(ticks,[bpm_periods[0]] + bpm_periods,'r+-')
    pylab.hold(True)
    pylab.plot([ticks[0],ticks[-1]],[bpm]*2,'g-')
    pylab.plot(rubato_start,[bpm]*len(rubato_start),'b+')
    pylab.plot(rubato_stop,[bpm]*len(rubato_stop),'b|')
    # ground truth
    if 'gt_ticks' in pool.descriptors.keys():
        gt_ticks  = pool.descriptors['gt_ticks']['values'][0]
        if len(gt_ticks) > 1:
            gt_bpm_periods = [60./(gt_ticks[i] - gt_ticks[i-1]) for i in range(1,len(gt_ticks))]
            p1 = pylab.plot(gt_ticks,[gt_bpm_periods[0]] + gt_bpm_periods,'rx:')
            p2 = pylab.plot(gt_ticks,[gt_bpm_periods[0]] + gt_bpm_periods,'rx:')
            #pylab.legend((p1[0],p2[0]),('Men','Women'))
    pylab.hold(False)
    pylab.show()

def parse_args():
    from optparse import OptionParser
    import sys
    usage = 'usage: %s [-v] <-i input_soundfile> [-g ground_truth_file]' % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option('-v','--verbose',
        action='store_true', dest='verbose', default=False,
        help='verbose mode')
    parser.add_option('-i','--input',
        action='store', dest='input_file', type='string',
        help='input file')
    parser.add_option('-o','--no-onsets',
        action='store_false', dest='use_onset', default=True,
        help='use onset features')
    parser.add_option('-b','--no-bands',
        action='store_false', dest='use_bands', default=True,
        help='use frequency bands features')
    parser.add_option('-p','--plot',
        action='store_true', dest='do_plots', default=False,
        help='plot each frame of features')
    parser.add_option('-P','--final-plot',
        action='store_true', dest='do_final_plots', default=False,
        help='plot the final tempo and beats outline')
    parser.add_option('-g','--ground-truth', default=None,
        action='store', dest='ground_truth_file', type='string',
        help='ground truth file')
    (options, args) = parser.parse_args()
    if options.input_file is None:
      print usage
      sys.exit(1)
    return options, args

if __name__ == '__main__':
    import sys, os.path, essentia
    options, args = parse_args()
    input_file = options.input_file

    # load audio file
    audio_file = essentia.AudioFileInput(filename = input_file)
    audio = audio_file()
    sampleRate = 44100.
    pool = essentia.Pool(input_file)

    if options.ground_truth_file is not None:
      import yaml
      if 'CLoader' in dir(yaml):
          load = lambda x: yaml.load( x, yaml.CLoader )
          load_all = lambda x: yaml.load_all( x, yaml.CLoader )
      else:
          load = yaml.load
          load_all = yaml.load_all
      if 'CDumper' in dir(yaml):
          dump = lambda x: yaml.dump( x,  Dumper=yaml.CDumper )
      else:
          dump = yaml.dump
      metadata = load(open(options.ground_truth_file))
      # add ground truth to pool
      if 'bpm' in metadata.keys():
          true_bpm = metadata['bpm']
          pool.add('gt_bpm', true_bpm, 0)
      #else:
      #    print 'no bpm found in ground truth file'
      if 'ticks' in metadata.keys():
          true_ticks = metadata['ticks']
          pool.add('gt_ticks', true_ticks, 0)
      #else:
      #    print 'no ticks found in ground truth file'

    compute(audio, pool, sampleRate = sampleRate, verbose = options.verbose,
      use_onset = options.use_onset,
      use_bands = options.use_bands,
      doplots = options.do_plots)
    if options.do_final_plots: plot_bpm_file(pool)
    if options.ground_truth_file is not None: print 'ground truth bpm', true_bpm
    if options.verbose:
      bpm    = pool.descriptors['tempotap_bpm']['values'][0]
      intervals = pool.descriptors['tempotap_intervals']['values'][0]
      bpm_periods = [60./interval for interval in intervals]
      ticks  = pool.descriptors['tempotap_ticks']['values'][0]
      rubato_start = pool.descriptors['tempotap_rubato_start']['values'][0]
      rubato_stop = pool.descriptors['tempotap_rubato_stop']['values'][0]
      print 'bpm', bpm
      print 'ticks', ticks
      print 'rubato_start', rubato_start
      print 'rubato_stop', rubato_stop
      print 'intervals', intervals

