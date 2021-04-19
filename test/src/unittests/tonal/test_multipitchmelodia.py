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


class TestMultiPitchMelodia(TestCase):

    def testZero(self):
        signal = zeros(1024)
        pitch = MultiPitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(MultiPitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(MultiPitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(MultiPitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(MultiPitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(MultiPitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'timeContinuity': -1})

    def testOnes(self):
        signal = ones(1024)
        pitch = MultiPitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testEmpty(self):
        pitch = MultiPitchMelodia()([])
        self.assertEqualVector(pitch, [])

    # FIXME-work in progress
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        pm = MultiPitchMelodia()
        pitch = pm(audio)
        print(type(pitch))
        print(type(array(pitch)))

        """
        #This code stores reference values in a file for later loading.
        save('multipitchmelodia.npy', array(pitch,dtype=object))             

        loadedMultiPitchMelodia = load(join(filedir(), 'pitchmelodia/multipitchmelodia.npy'))
        expectedMultiPitchMelodia = loadedMultiPitchMelodia.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedMultiPitchMelodia, 2)

                # This code stores reference values in a file for later loading.
        save('multipitchmelodia.npy', array(pitch))
        # Reference samples are loaded as expected values
        expected_multipitchmelodia_npy = load(join(filedir(), 'pitchmelodia/multipitchmelodia.npy'))

        # Loop through all Melframes to regression test each one against the file reference values.
        index = 0
        while index<len(expected_multipitchmelodia_npy):
           self.assertAlmostEqualVectorFixedPrecision(expected_multipitchmelodia_npy[index], pitch[index], 8)
           index+=1
        """
suite = allTests(TestMultiPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
