#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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



from numpy import *
from essentia_test import *

framesize = 1024
hopsize = 512


class TestOnsetDetection(TestCase):

    def testZero(self):
        # Inputting zeros should return no onsets (empty array)
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/techno_loop.wav'), sampleRate = 44100)()
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        win = Windowing(type='hamming')
        fft = FFT()
        onset_hfc = OnsetDetection(method='hfc')
        onset_complex = OnsetDetection(method='complex')
        onset_complex_phase = OnsetDetection(method='complex_phase')
        onset_melflux = OnsetDetection(method='melflux')
        onset_flux = OnsetDetection(method='flux')
        onset_rms = OnsetDetection(method='rms')
        for frame in frames:
            fft_frame = fft(win(frame))
            mag, ph = CartesianToPolar()(fft_frame)
            mag = zeros(len(mag))
            self.assertEqual(onset_hfc(mag, ph), 0)
            self.assertEqual(onset_complex(mag, ph), 0)
            self.assertEqual(onset_complex_phase(mag, ph), 0)
            self.assertEqual(onset_melflux(mag, ph), 0)
            self.assertEqual(onset_flux(mag, ph), 0)
            self.assertEqual(onset_rms(mag, ph), 0)

    def testImpulse(self):
        # tests that for an impulse will yield the correct position
        audiosize = 10000
        audio = zeros(audiosize)
        pos = 5.5  # impulse will be in between frames 4 and 5
        audio[int(floor(pos*(hopsize)))] = 1.
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True)
        win = Windowing(type='hamming', zeroPadding=framesize)
        fft = FFT()
        onset_hfc = OnsetDetection(method='hfc')
        onset_complex_phase = OnsetDetection(method='complex_phase')
        onset_rms = OnsetDetection(method='rms')
        onset_flux = OnsetDetection(method='flux')
        onset_melflux = OnsetDetection(method='melflux')
        onset_complex = OnsetDetection(method='complex')

        nframe = 0
        for frame in frames:
            mag, ph = CartesianToPolar()(fft(win(frame)))

            # 'rms' (energy flux) and 'melflux' method will result in a non-zero value on frames 4 and 5,
            # energy flux for frame 6 is zero due to half-rectification
            # 'flux' on contrary will results in non-zero value for frame 6, as it does not half-rectify

            if nframe == floor(pos)-1:  # 4th frame
                self.assertNotEqual(onset_complex_phase(mag, ph), 0)
                self.assertNotEqual(onset_hfc(mag, ph), 0)
                self.assertNotEqual(onset_rms(mag, ph), 0)
                self.assertNotEqual(onset_flux(mag, ph), 0)
                self.assertNotEqual(onset_melflux(mag, ph), 0)
                self.assertNotEqual(onset_complex(mag, ph), 0)
            elif nframe == ceil(pos)-1:  # 5th frame
                self.assertNotEqual(onset_complex_phase(mag, ph), 0)
                self.assertNotEqual(onset_hfc(mag, ph), 0)
                self.assertNotEqual(onset_rms(mag, ph), 0)
                self.assertNotEqual(onset_flux(mag, ph), 0)
                self.assertNotEqual(onset_melflux(mag, ph), 0)
                self.assertNotEqual(onset_complex(mag, ph), 0)
            elif nframe == ceil(pos):  # 6th frame
                self.assertEqual(onset_complex_phase(mag, ph), 0)
                self.assertEqual(onset_hfc(mag, ph), 0)
                self.assertEqual(onset_rms(mag, ph), 0)
                self.assertNotEqual(onset_flux(mag, ph), 0)
                self.assertEqual(onset_melflux(mag, ph), 0)
                self.assertNotEqual(onset_complex(mag, ph), 0)
            else:
                self.assertEqual(onset_complex_phase(mag,ph), 0)
                self.assertEqual(onset_hfc(mag, ph), 0)
                self.assertEqual(onset_rms(mag, ph), 0)
                self.assertEqual(onset_flux(mag, ph), 0)
                self.assertEqual(onset_melflux(mag, ph), 0)
                self.assertEqual(onset_complex(mag, ph), 0)
            nframe += 1

    def testConstantInput(self):
        audio = ones(44100*5)
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        win = Windowing(type='hamming')
        fft = FFT()
        onset_hfc = OnsetDetection(method='hfc')
        onset_complex = OnsetDetection(method='complex')
        onset_flux = OnsetDetection(method='flux')
        onset_melflux = OnsetDetection(method='melflux')
        onset_rms = OnsetDetection(method='rms')
        onset_complex_phase = OnsetDetection(method='complex_phase')
        found_complex = []
        found_hfc = []
        found_flux = []
        found_melflux = []
        found_rms = []
        found_complex_phase = []
        for frame in frames:
            fft_frame = fft(win(frame))
            mag, ph = CartesianToPolar()(fft_frame)
            mag = zeros(len(mag))
            found_hfc += [onset_hfc(mag, ph)]
            found_complex += [onset_complex(mag, ph)]
            found_flux +=  [onset_flux(mag, ph)]
            found_melflux += [onset_melflux(mag, ph)]
            found_rms += [onset_rms(mag, ph)]
            found_complex_phase += [onset_complex_phase(mag, ph)]
        self.assertEqualVector(found_complex, zeros(len(found_complex)))
        self.assertEqualVector(found_hfc, zeros(len(found_hfc)))
        self.assertEqualVector(found_flux, zeros(len(found_flux)))
        self.assertEqualVector(found_melflux, zeros(len(found_melflux)))
        self.assertEqualVector(found_rms, zeros(len(found_rms)))
        self.assertEqualVector(found_hfc, zeros(len(found_hfc)))
        self.assertEqualVector(found_complex_phase, zeros(len(found_complex_phase)))
    
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetection(), { 'sampleRate':-1 })
        self.assertConfigureFails(OnsetDetection(), { 'method':'unknown' })

    def testComplexInputSizeMismatch(self):
        spectrum = ones(1024)
        phase = ones(512)
        self.assertComputeFails(OnsetDetection(method='complex'), spectrum, phase)
        self.assertComputeFails(OnsetDetection(method='complex_phase'), spectrum, phase)

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/techno_loop.wav'), sampleRate = 44100)()
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        win = Windowing(type='hamming')
        fft = FFT()
        onset_hfc = OnsetDetection(method='hfc')
        onset_complex = OnsetDetection(method='complex')
        onset_complex_phase = OnsetDetection(method='complex_phase')
        onset_melflux = OnsetDetection(method='melflux')
        onset_flux = OnsetDetection(method='flux')
        onset_rms = OnsetDetection(method='rms')

        hfc_out = []
        complex_out = []
        complex_phase_out = []
        melflux_out = []
        flux_out = []
        rms_out = []

        for frame in frames:
            fft_frame = fft(win(frame))
            mag, ph = CartesianToPolar()(fft_frame)
            hfc_out += [onset_hfc(mag, ph)]
            complex_out += [onset_complex(mag, ph)]
            complex_phase_out+= [onset_complex_phase(mag, ph)]
            melflux_out += [onset_melflux(mag, ph)]
            flux_out += [onset_flux(mag, ph)]
            rms_out += [onset_rms(mag, ph)]

        """
        This code was used to obtain reference samples for storing in a file.
        save('hfc.npy', hfc_out)
        save('complex.npy', complex_out)
        save('complex_phase.npy', complex_phase_out)
        save('melflux.npy', melflux_out)
        save('flux.npy', flux_out)                
        save('rms.npy', rms_out)                
        """
        
        # Reference samples are loaded as expected values
        expected_hfc = load(join(filedir(), 'onsetdetection/hfc.npy'))
        expected_complex =  load(join(filedir(), 'onsetdetection/complex.npy'))
        expected_complex_phase =  load(join(filedir(), 'onsetdetection/complex_phase.npy'))
        expected_melflux = load(join(filedir(), 'onsetdetection/melflux.npy'))
        expected_flux =  load(join(filedir(), 'onsetdetection/flux.npy'))
        expected_rms =  load(join(filedir(), 'onsetdetection/rms.npy'))

        self.assertAlmostEqualVectorFixedPrecision(hfc_out, expected_hfc,1)
        self.assertAlmostEqualVectorFixedPrecision(complex_out, expected_complex,4)
        self.assertAlmostEqualVectorFixedPrecision(complex_phase_out, expected_complex_phase,4)
        self.assertAlmostEqualVectorFixedPrecision(melflux_out, expected_melflux,2)
        self.assertAlmostEqualVectorFixedPrecision(flux_out, expected_flux,4)
        self.assertAlmostEqualVectorFixedPrecision(rms_out, expected_rms,4)


suite = allTests(TestOnsetDetection)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
