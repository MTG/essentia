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

#! /usr/bin/env python

import sys, os
import essentia, essentia.standard, essentia.streaming, essentia.translate
import essentia.utils as utils
from essentia.streaming import *

namespace = 'lowlevel'

lowlevelFrameSize = 2048
lowlevelHopSize = 1024
analysisSampleRate = 44100.0
minFrequency = analysisSampleRate/lowlevelFrameSize

class LowLevelSpectralExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=lowlevelFrameSize,
                       hopSize=lowlevelHopSize,
                       halfSampleRate=analysisSampleRate/2.,
                       minFrequency=minFrequency):
        super(LowLevelSpectralExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')

        # silence rate:
        thresholds=[utils.db2lin(val/2.0) for val in [-20.0, -30.0, -60.0]]
        sr = SilenceRate(thresholds=thresholds)
        fc.frame >> sr.frame
        # windowing:
        w = Windowing(type='blackmanharris62')
        fc.frame >> w.frame
        # spectrum:
        spec = Spectrum()
        w.frame >> spec.frame
        # temporal descriptors
        zcr = ZeroCrossingRate()
        fc.frame >> zcr.signal
        # mfcc:
        mfcc = MFCC()
        spec.spectrum >> mfcc.spectrum
        mfcc.bands >> None
        # spectral decrease:
        square = UnaryOperator(type='square')
        decrease = Decrease(range=halfSampleRate)
        spec.spectrum >> square.array >> decrease.array
        # spectral energy:
        energy = Energy()
        spec.spectrum >> energy.array
        # spectral energy ratio:
        ebr_low = EnergyBand(startCutoffFrequency=20,
                             stopCutoffFrequency=150)
        ebr_mid_low = EnergyBand(startCutoffFrequency=150,
                                 stopCutoffFrequency=800)
        ebr_mid_hi = EnergyBand(startCutoffFrequency=800,
                                stopCutoffFrequency=4000)
        ebr_hi = EnergyBand(startCutoffFrequency=4000,
                            stopCutoffFrequency=20000)

        spec.spectrum >> ebr_low.spectrum
        spec.spectrum >> ebr_mid_low.spectrum
        spec.spectrum >> ebr_mid_hi.spectrum
        spec.spectrum >> ebr_hi.spectrum
        # spectral hfc:
        hfc = HFC()
        spec.spectrum >> hfc.spectrum
        # spectral rms:
        rms = RMS()
        spec.spectrum >> rms.array
        # spectral flux:
        flux = Flux()
        spec.spectrum >> flux.spectrum
        # spectral roll off:
        ro = RollOff()
        spec.spectrum >> ro.spectrum
        # spectral strong peak:
        sp = StrongPeak()
        spec.spectrum >> sp.spectrum
        # bark bands:
        barkBands = BarkBands(numberBands=27)
        spec.spectrum >> barkBands.spectrum
        # spectral crest:
        crest = Crest()
        barkBands.bands >> crest.array
        # spectral flatness db:
        flatness = FlatnessDB()
        barkBands.bands >> flatness.array
        # spectral barkbands central moments:
        cm = CentralMoments(range=26) # BarkBands::numberBands - 1
        ds = DistributionShape()
        barkBands.bands >> cm.array
        cm.centralMoments >> ds.centralMoments
        # spectral complexity:
        tc = SpectralComplexity(magnitudeThreshold=0.005)
        spec.spectrum >> tc.spectrum
        # pitch detection:
        pitch = PitchDetection(frameSize=frameSize)
        spec.spectrum >> pitch.spectrum
        # pitch salience:
        ps = PitchSalience()
        spec.spectrum >> ps.spectrum
        # harmonic peaks:
        peaks = SpectralPeaks(orderBy='frequency',
                              minFrequency=minFrequency)
        harmPeaks = HarmonicPeaks()
        odd2even = OddToEvenHarmonicEnergyRatio()
        tristimulus = Tristimulus()
        inharmonicity = Inharmonicity()

        spec.spectrum >> peaks.spectrum
        peaks.frequencies >> harmPeaks.frequencies
        peaks.magnitudes >>  harmPeaks.magnitudes
        pitch.pitch >> harmPeaks.pitch
        harmPeaks.harmonicFrequencies >> tristimulus.frequencies
        harmPeaks.harmonicMagnitudes  >> tristimulus.magnitudes
        harmPeaks.harmonicFrequencies >> odd2even.frequencies
        harmPeaks.harmonicMagnitudes  >> odd2even.magnitudes
        harmPeaks.harmonicFrequencies >> inharmonicity.frequencies
        harmPeaks.harmonicMagnitudes  >> inharmonicity.magnitudes

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        # silence rate:
        self.outputs['silence_rate_20dB'] = sr.threshold_0
        self.outputs['silence_rate_30dB'] = sr.threshold_1
        self.outputs['silence_rate_60dB'] = sr.threshold_2
        # zero crossing rate:
        self.outputs['zerocrossingrate'] = zcr.zeroCrossingRate
        # MFCC rate:
        self.outputs['mfcc'] = mfcc.mfcc
        # spectral decrease:
        self.outputs['spectral_decrease'] = decrease.decrease
        # spectral energy:
        self.outputs['spectral_energy'] = energy.energy
        # spectral energy ratio:
        self.outputs['spectral_energyband_low'] = ebr_low.energyBand
        self.outputs['spectral_energyband_middle_low'] = ebr_mid_low.energyBand
        self.outputs['spectral_energyband_middle_high'] = ebr_mid_hi.energyBand
        self.outputs['spectral_energyband_high'] = ebr_hi.energyBand
        # spectral hfc:
        self.outputs['hfc'] = hfc.hfc
        # spectral rms:
        self.outputs['spectral_rms'] = rms.rms
        # spectral flux:
        self.outputs['spectral_flux'] = flux.flux
        # spectral roll off:
        self.outputs['spectral_rolloff'] = ro.rollOff
        # spectral strong peak:
        self.outputs['spectral_strongpeak'] = sp.strongPeak
        # bark bands:
        self.outputs['barkbands'] = barkBands.bands
        # spectral crest:
        self.outputs['spectral_crest'] = crest.crest
        # spectral flatness db:
        self.outputs['spectral_flatness_db'] = flatness.flatnessDB
        # spectral barkbands central moments:
        self.outputs['barkbands_kurtosis'] = ds.kurtosis
        self.outputs['barkbands_spread'] = ds.spread
        self.outputs['barkbands_skewness'] = ds.skewness
        # harmonic peaks:
        self.outputs['spectral_complexity'] = tc.spectralComplexity
        self.outputs['pitch'] = pitch.pitch
        self.outputs['pitch_instantaneous_confidence'] = pitch.pitchConfidence
        self.outputs['pitch_salience'] = ps.pitchSalience
        self.outputs['inharmonicity'] = inharmonicity.inharmonicity
        self.outputs['oddtoevenharmonicenergyratio'] = odd2even.oddToEvenHarmonicEnergyRatio
        self.outputs['tristimulus'] = tristimulus.tristimulus


usage = 'lowlevelspectral.py [options] <inputfilename> <outputfilename>'

def parse_args():

    import numpy
    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=usage, version=essentia_version)

    parser.add_option("-c","--cpp", action="store_true", dest="generate_cpp",
      help="generate cpp code from CompositeBase algorithm")

    parser.add_option("-d", "--dot", action="store_true", dest="generate_dot",
      help="generate dot and cpp code from CompositeBase algorithm")

    (options, args) = parser.parse_args()

    return options, args



if __name__ == '__main__':

    opts, args = parse_args()

    if len(args) != 2:
        cmd = './'+os.path.basename(sys.argv[0])+ ' -h'
        os.system(cmd)
        sys.exit(1)

    if opts.generate_dot:
        essentia.translate(LowLevelSpectralExtractor, 'streaming_extractorlowlevelspectral', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(LowLevelSpectralExtractor, 'streaming_extractorlowlevelspectral', dot_graph=False)


    loader = MonoLoader(filename=args[0])
    lowlevelExtractor = LowLevelSpectralExtractor()
    pool = essentia.Pool()

    loader.audio >> lowlevelExtractor.signal

    for desc, output in lowlevelExtractor.outputs.items():
        output >> (pool, namespace + '.'+desc)

    essentia.run(loader)

    # compute aggregation on values:
    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    exceptions = {'lowlevel.mfcc' : ['mean', 'cov', 'icov']}
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats,
                                                 exceptions=exceptions)(pool)

    essentia.standard.YamlOutput(filename=args[1])(poolStats)
