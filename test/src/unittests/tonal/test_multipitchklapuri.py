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


class TestMultiPitchKlapuri(TestCase):

    def testZero(self):
        signal = zeros(1024)
        pitch = MultiPitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(MultiPitchKlapuri(), {'binResolution': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'frameSize': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'harmonicWeight': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'hopSize': -1})        
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeCompression': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeCompression': 2})
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'maxFrequency': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'minFrequency': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'numberHarmonics': -1})            
        self.assertConfigureFails(MultiPitchKlapuri(), {'referenceFrequency': -1})             
        self.assertConfigureFails(MultiPitchKlapuri(), {'sampleRate': -1})

    def testOnes(self):
        # FIXME. Need to derive a rational why this output occurs for a constant input
        signal = ones(1024)
        pitch = MultiPitchKlapuri()(signal)
        print(pitch)
        print(len(pitch))
        expectedPitch= [[ 92.498886, 184.99854 ],
            [108.110695, 151.1358  ],
            [108.73698,  151.1358  ],
            [108.73698,  151.1358  ],
            [108.73698,  151.1358  ],
            [108.110695, 151.1358  ],
            [ 92.498886, 184.99854 ]]

        index=0
        while (index<len(expectedPitch)):
            self.assertAlmostEqualVector(pitch[index], expectedPitch[index],8)
            index+=1

    def testEmpty(self):
        pitch = MultiPitchKlapuri()([])
        self.assertEqualVector(pitch, [])


    # FIXME-work in progress
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        pm = MultiPitchKlapuri()
        pitch = pm(audio)
        print(pitch[0])
        save('klapuri.npy',pitch)

        # Reference samples are loaded as expected values
        expected_klapuri_npy = load(join(filedir(), 'pitchmelodia/klapuri.npy'))



        index=0
        while (index<len(expected_klapuri_npy)):
            print(pitch[index])
            #self.assertAlmostEqualVector(pitch[index], expected_klapuri_npy[index],8)
            index+=1


suite = allTests(TestMultiPitchKlapuri)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
