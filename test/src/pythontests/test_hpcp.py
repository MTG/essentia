#!/usr/bin/env python

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



import numpy
import essentia
import sys

hpcp = essentia.HPCP(size = 36,
                     referenceFrequency = 440.0,
                     bandPreset = False,
                     minFrequency = 40.0,
                     maxFrequency = 5000.0,
                     midLowFrequency = 500.0,
                     midHighFrequency = 1000.0,
                     useWeight = True,
                     squareWeight = True,
                     nonLinear = False,
                     windowSize = 4.0/3.0,
                     sampleRate = 44100.0)

frequencies = [947.3032226562, 1399.7023925781, 1216.341796875, 762.8428344727,
               654.5443115234, 520.2301635742, 4575.2436523438, 409.2276306152,
               4791.2998046875, 2573.521484375, 827.6868286133, 3270.7573242188,
               1922.7291259766, 3627.2180175781, 3452.2028808594, 1671.4516601562,
               1588.4647216797, 4390.6713867188, 3758.7380371094, 3160.9790039062,
               3866.9069824219, 2903.2268066406, 3974.7385253906, 2157.0363769531,
               4211.8017578125, 4906.0146484375, 4107.1103515625, 188.6952819824,
               115.285446167, 2071.6237792969, 2310.9438476562, 3019.662109375,
               4281.8125, 2740.8828125]

magnitudes = [0.0003832988, 0.000289413, 0.0002247146, 0.0001418052,
              0.0001305008, 0.0001210179, 0.0001141675, 0.0000843476,
              0.0000809668, 0.0000756415, 0.0000697662, 0.0000677788,
              0.0000631657, 0.0000616424, 0.0000587001, 0.0000538562,
              0.0000514311, 0.0000506943, 0.0000472046, 0.0000441288,
              0.000041539,  0.0000410511, 0.0000391758, 0.0000388247,
              0.0000357574, 0.0000334767, 0.000031703, 0.000030963,
              0.0000299147, 0.0000292564, 0.0000250094, 0.0000204391,
              0.000018821, 0.0000168285]

hpcp_value = hpcp(frequencies, magnitudes)
print hpcp_value
