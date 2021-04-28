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

def create_python_algorithms(essentia):

    '''
    # Spectral Decrease
    class SpectralDecrease(essentia.Decrease):
        def configure(self, sampleRate = 44100, **kwargs):
            essentia.Decrease.configure(self, range = sampleRate*0.5)

    setattr(essentia, 'SpectralDecrease', SpectralDecrease)


    # AudioDecrease
    class AudioDecrease(essentia.Decrease):
        def configure(self, blockSize, sampleRate = 44100, **kwargs):
            essentia.Decrease.configure(self, range = (blockSize-1.0)/sampleRate)

    setattr(essentia, 'AudioDecrease', AudioDecrease)


    # SpectralCentroid
    class SpectralCentroid(essentia.Centroid):
        def configure(self, sampleRate = 44100, **kwargs):
            essentia.Centroid.configure(self, range = sampleRate*0.5)

    setattr(essentia, 'SpectralCentroid', SpectralCentroid)


    # AudioCentroid
    class AudioCentroid(essentia.Centroid):
        def configure(self, blockSize, sampleRate = 44100, **kwargs):
            essentia.Centroid.configure(self, range = (blockSize-1.0)/sampleRate)

    setattr(essentia, 'AudioCentroid', AudioCentroid)


    # SpectralCentralMoments
    class SpectralCentralMoments(essentia.CentralMoments):
        def configure(self, sampleRate = 44100, **kwargs):
            essentia.CentralMoments.configure(self, range = sampleRate*0.5)

    setattr(essentia, 'SpectralCentralMoments', SpectralCentralMoments)


    # AudioCentralMoments
    class AudioCentralMoments(essentia.CentralMoments):
        def configure(self, blockSize, sampleRate = 44100, **kwargs):
            essentia.CentralMoments.configure(self, range = (blockSize-1.0)/sampleRate)

    setattr(essentia, 'AudioCentralMoments', AudioCentralMoments)
    '''

    default_fc = essentia.FrameCutter()



    # FrameGenerator
    class FrameGenerator(object):
        __struct__ = { 'name': 'FrameGenerator',
                       'category': 'Standard',
                       'inputs': [],
                       'outputs': [],
                       'parameters': default_fc.__struct__['parameters'],
                       'description': '''The FrameGenerator is a Python generator for the FrameCutter algorithm. It is not available in C++.

FrameGenerator inherits all the parameters of the FrameCutter. The way to use it in Python is the following:

  for frame in FrameGenerator(audio, frameSize, hopSize):
      do_something()

''' }


        def __init__(self, audio, frameSize = default_fc.paramValue('frameSize'), 
                                  hopSize = default_fc.paramValue('hopSize'),
                                  startFromZero = default_fc.paramValue('startFromZero'),
                                  validFrameThresholdRatio = default_fc.paramValue('validFrameThresholdRatio'),
                                  lastFrameToEndOfFile=default_fc.paramValue('lastFrameToEndOfFile')):

            self.audio = audio
            self.frameSize = frameSize
            self.hopSize = hopSize
            self.startFromZero = startFromZero
            self.validFrameThresholdRatio = validFrameThresholdRatio
            self.lastFrameToEndOfFile=lastFrameToEndOfFile
            self.frame_creator = essentia.FrameCutter(frameSize = frameSize,
                                                       hopSize = hopSize,
                                                       startFromZero = startFromZero,
                                                       validFrameThresholdRatio=validFrameThresholdRatio,
                                                       lastFrameToEndOfFile=lastFrameToEndOfFile)

        def __iter__(self):
            return self

        def __next__(self):
            frame = self.frame_creator.compute(self.audio)
            if frame.size == 0:
                raise StopIteration
            else:
                return frame

        next = __next__  # Python 2

        def num_frames(self):
            if self.startFromZero:
                if not self.lastFrameToEndOfFile:
                    size = int(round(len(self.audio) / float(self.hopSize)))
                else:
                    size = int(round((len(self.audio)+self.frameSize) / float(self.hopSize)))
            else:
                size = int(round((len(self.audio)+self.frameSize/2.0)/float(self.hopSize)))
            return size

        def frame_times(self, sampleRate):
            times = []
            if self.startFromZero:
                start = self.frameSize/2.0
            else:
                start = 0.

            for i in range(self.num_frames()):
                times.append((start + self.hopSize * i) / sampleRate)
            return times


    setattr(essentia, 'FrameGenerator', FrameGenerator)
