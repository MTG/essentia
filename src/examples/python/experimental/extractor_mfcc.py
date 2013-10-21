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


# TODO probably this code is completely outdated for Essentia 2.0, storing this code just in case


import sys
import essentia, essentia.streaming, essentia.standard

# When creating your own composite algorithm, your class must inherit from
# essentia.streaming.CompositeBase.
class ExtractorMfcc(essentia.streaming.CompositeBase):

    # Specify the parameters of your algorithm and their default values as inputs to the __init__
    # method. You can then use these parameters to configure the inner algorithms.
    #
    # Note: If you desire to translate your python composite algorithm to c++ code, you may not
    # perform any operations other than those described below in the __init__ method. This is because
    # the translator does not support translating anything other than creating algorithms, declaring
    # inputs and outputs, and connecting inner algorithms. You do not need to abide by this
    # restriction if you never intend to translate your composite algorithm to c++ code.
    #
    # To make this point clearer: it might have been convenient to only accept a frameSize
    # parameter, and internally, set a hopSize variable to half the frameSize. So within our
    # __init__ method, we would have a statement like "hopSize = frameSize/2". However, because this
    # statement is not directly one of the operations described above, it will not be translated and
    # an error will be raised instead.
    def __init__(self, frameSize=2048, hopSize=1024, windowType='blackmanharris62'):
        # Don't forget to call the base class's __init__ method!
        super(ExtractorMfcc, self).__init__()

        # Create and configure each inner algorithm
        fc = essentia.streaming.FrameCutter(frameSize=frameSize,
                                            hopSize=hopSize,
                                            silentFrames='noise')
        wnd = essentia.streaming.Windowing(type=windowType)
        spec = essentia.streaming.Spectrum()
        mfcc = essentia.streaming.MFCC()

        # Declare the inputs of your composite algorithm in the self.inputs dictionary. The keys of
        # this dictionary should be the name you give your input, and the values are the inputs of
        # inner algorithms
        self.inputs['audio'] = fc.signal

        # Make connections between the inner algorithms
        fc.frame >> wnd.frame >> spec.frame
        spec.spectrum >> mfcc.spectrum

        # If an output is not needed, it still must be connected--connect it to None
        mfcc.bands >> None

        # Declare your outputs in the same way as the inputs. Output names are allowed to be the
        # same as input names, and is encouraged, if it makes sense for your composite algorithm.
        # If the names match, you can do things like chaining connections as we did for the
        # Windowing algorithm above.
        self.outputs['mfcc'] = mfcc.mfcc


if __name__ == '__main__':
    # Make sure the command was well-formed.
    if len(sys.argv) < 3:
        print 'Usage: extractor_mfcc.py <input audio filename> <output yaml filename>'
        sys.exit(1)

    # Loaders must be specified outside your composite algorithm.
    loader = essentia.streaming.MonoLoader(filename=sys.argv[1])

    # We are using the default values of our parameters so we don't specify any keyword arguments.
    mfccex = ExtractorMfcc()

    p = essentia.Pool()

    # When connecting to/from your composite algorithm, use the names you declared in the
    # self.inputs and self.outputs dictionaries, respectively.
    loader.audio >> mfccex.audio
    mfccex.mfcc >> (p, 'mfcc')

    essentia.run(loader)

    # CompoxiteBase algorithms can be translated into c++ code and dot graphs
    # can also be generated:
    essentia.translate(ExtractorMfcc,     # algorithm to be translated
                       'myExtractorMfcc', # output name for the c++ and dot generated files
                       dot_graph=True)    # whether dot file should be generated
    essentia.standard.YamlOutput(filename=sys.argv[2])(p)
