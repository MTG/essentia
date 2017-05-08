#!/usr/bin/env python

# Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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



from essentia_test import *


class TestChromaprinter(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Chromaprinter(), [])


    def testRegression(self):
        #  the expected finguerprint was computed using the pyacousticid python module: https://pypi.python.org/pypi/pyacoustid 
        expected = 'AQAA3ZnWRIom4Y9xEe9RH_qHbx5sCi98XQj34z6a4MI_HPmL5BZ-OFd29MK-kDHeCw2zF-WLH4ryM-gb41safLjS4zmSWUk-hEkTK8GVG29weFnWxHiP5KnkItx0NMeHX7gy3XiaTETSJ8hz-Dmx45NQ58Hho79wcsgljcUkbgz8yYF-5MV_dHqOM8mJf_jRPKiqfcQnC_dxBQmpRgvy7MRPbDeeR_gGhg-a_EkC-Tqu7XjS4Aum8WiDST-aPxoy6FsmPKRS-MFN4w90PfArnMaVvqi14B6OS_AB__Bx5oK_PMFxzDJ8ou3wCiolvDmQW8GXTgl-NOR4TO_xX4gJLkyWQLmUEH8QhtlhH_3xPfiiL8Xx7EbVwu_QD6cOPQx2vOPxRmBeUD_KX3i248vRI4yTHToX40f1xHCTxtCjJ9B2sIiE5gp-_BLapPCeC0f5C_8j_CQuPDk6j2geHhdeWcgf6Ed0BvtR8mh07DlyKYeydGyC_IKf4qKP58d_OF1xnMlxPcFF5nCPB70Q5jyOCxrS73hh9XiF4zdK8Th-uE5RPZfQOBOPtcnh-MKzIl9SaOkRPsiVpXgk6Mxx-WicKwgHIhAgDihEBSBAARGBRYqBIAgGQhgCrAAKKaSIAUJZLQyRgDxDhIESAMEsQsoIAACTCgAEBAAMOYKEAAoRYgFCCAkAmIBAIgAIMEQBphBRjgBhHbMIKGEEUGoQRSRBQBDEABAQACYkUkAIgQAhUkiCLANECcOIEwlBIBRwhDokBNGCGSCYIIgQBZ0ywBkAqDPWEAMAUMYQYRBpABkAoJgCmDAAQwQYISh1AAAiiIDWKIeAMQA'
        
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'mozart_c_major_30sec.wav'))()
        result =  Chromaprinter()(audio)
        
        self.assertEqualVector(expected, result)

    def testInvalidParam(self):
        self.assertConfigureFails(Chromaprinter(), { 'maxLength': -1 })
        self.assertConfigureFails(Chromaprinter(), { 'sampleRate': -1 })

    def testMaxLength(self):
     
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'mozart_c_major_30sec.wav'))()
        result1 =  Chromaprinter()(audio)
        result2 =  Chromaprinter(maxLength=50)(audio)
        
        self.assertEqualVector(result1, result2)


suite = allTests(TestChromaprinter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
