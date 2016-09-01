/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_AUDIOADAPTOR_H
#define ESSENTIA_AUDIOADAPTOR_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class FrameToReal : public Algorithm {

 private:


  Input<std::vector<Real> > _frames;
  Output<std::vector<Real> > _audio;



//    Output<std::vector<AudioSample> > _audio; // ?? check what data type is appropriate
  int _frameSize;
  int _hopSize;


 public:
  FrameToReal() {
    declareInput(_frames, "signal", "the input audio frame");
    //declareOutput(_frame, "frame", "the output overlap-add audio signal frame");
    declareOutput(_audio, "signal", "the output audio samples"); // type should be signal to be consistent??

  }

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing the overlap-add process", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the overlap-add function is computed", "(0,inf)", 128);

  }
  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FrameToReal : public StreamingAlgorithmWrapper {
//class FrameToReal : public Algorithm {

 protected:

//  Sink<std::vector<Real> > _frames; // input
//  Source<Real> _audio; // output

  Sink<std::vector<Real> > _frames; // input
  Source<Real> _audio; // output




  int _frameSize;
  int _hopSize;


  bool _configured;


 public:
FrameToReal() {
    declareAlgorithm("FrameToReal");
    //declareInput(_windowedFrame, TOKEN, "frame");

  declareInput(_frames, TOKEN,"signal");  // TODO: update when algorithmwrapper is fixed
  declareOutput(_audio, TOKEN, "signal");


    _audio.setBufferType(BufferUsage::forLargeAudioStream); // TODO: check this

  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ZEROCROSSINGRATE_H
