/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_FRAMECUTTER_H
#define ESSENTIA_FRAMECUTTER_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class FrameCutter : public Algorithm {

 protected:
  Input<std::vector<Real> > _buffer;
  Output<std::vector<Real> > _frame;

  bool _startFromZero;
  bool _lastFrameToEndOfFile;
  int _startIndex;
  int _frameSize;
  int _hopSize;
  bool _lastFrame;
  int _validFrameThreshold;


 public:
   FrameCutter() : _startIndex(0) {
    declareInput(_buffer, "signal", "the buffer from which to read data");
    declareOutput(_frame, "frame", "the frame to write to");
  }

  void declareParameters() {
    declareParameter("frameSize", "the output frame size", "[1,inf)", 1024);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 512);
    declareParameter("validFrameThresholdRatio", "frames smaller than this "
                     "ratio will be discarded, those larger will be "
                     "zero-padded to make a full frame (i.e. a value of 0 will "
                     "never discard frames and a value of 1 will only keep "
                     "frames that are of length 'frameSize')",
                     "[0,1]", 0.);
    declareParameter("startFromZero", "whether to start the first frame at "
                     "time 0 if true, or -frameSize/2 otherwise (zero-centered)",
                     "{true,false}", false);
    declareParameter("lastFrameToEndOfFile", "whether the beginning of the last "
                     "frame should reach the end of file. Only applicable if "
                     "startFromZero is true",
                     "{true,false}", false);
  }

  void configure();
  void reset();

  void compute();

  static const char* name;
  static const char* description;


};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {


class FrameCutter : public Algorithm {
 protected:

  Sink<AudioSample> _audio;
  Source<std::vector<AudioSample> > _frames;

  int _frameSize;
  int _hopSize;
  int _startIndex; // the desired start index of the next frame
  int _streamIndex; // the index in the stream

  int _validFrameThreshold;
  bool _startFromZero;
  bool _lastFrameToEndOfFile;

  enum SilenceType {KEEP, DROP, ADD_NOISE};
  SilenceType typeFromString(const std::string& name) const;
  standard::Algorithm * _noiseAdder;

  SilenceType _silentFrames;


 public:
  FrameCutter() {
    // at the beginning, releaseSize is set to 0, but will become hopSize once
    // we are done zero-padding the signal
    declareInput(_audio, _frameSize, 0, "signal", "the input audio signal");
    declareOutput(_frames, 1, "frame", "the frames of the audio signal");
    _noiseAdder = standard::AlgorithmFactory::create("NoiseAdder");
  }
  ~FrameCutter() {
    delete _noiseAdder;
  }

  void declareParameters() {
    declareParameter("frameSize", "the size of the frame to cut", "[1,inf)", 1024);
    declareParameter("hopSize", "the number of samples to jump after a frame is output", "[1,inf)", 512);
    declareParameter("silentFrames", "whether to [keep/drop/add noise to] silent frames", "{drop,keep,noise}", "noise");
    declareParameter("validFrameThresholdRatio", "frames smaller than this "
                     "ratio will be discarded, those larger will be "
                     "zero-padded to make a full frame (i.e. a value of 0 will "
                     "never discard frames and a value of 1 will only keep "
                     "frames that are of length 'frameSize')",
                     "[0,1]", 0.);
    declareParameter("startFromZero", "whether to start the first frame at "
                     "time 0 if true, or -frameSize/2 otherwise (zero-centered)",
                     "{true,false}",
                     false);
    declareParameter("lastFrameToEndOfFile", "whether the beginning of the last "
                     "frame should reach the end of file. Only applicable if "
                     "startFromZero is true",
                     "{true,false}", false);
  }

  void reset();
  void configure();
  AlgorithmStatus process();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FRAMECUTTER_H
