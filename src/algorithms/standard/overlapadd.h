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

#ifndef ESSENTIA_OVERLAPADD_H
#define ESSENTIA_OVERLAPADD_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class OverlapAdd : public Algorithm {

 private:

  Input<std::vector<Real> > _windowedFrame;
  Output<std::vector<Real> > _output;

  int _frameSize;
  int _hopSize;
  Real _gain;
  float _normalizationGain;
  std::vector<Real> _frameHistory;
  std::vector<Real> _tmpFrame;

 public:
  OverlapAdd() {
    declareInput(_windowedFrame, "signal", "the windowed input audio frame");
    declareOutput(_output, "signal", "the output overlap-add audio signal frame");
  }

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing the overlap-add process", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the overlap-add function is computed", "(0,inf)", 128);
    declareParameter("gain", "the normalization gain that scales the output signal. Useful for IFFT output", "(0.,inf)", 1.);
  }
  void compute();
  void configure();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class OverlapAdd : public Algorithm {

 protected:
  Sink<std::vector<Real> > _frames;
  Source<Real> _output;

  int _frameSize;
  int _hopSize;
  Real _gain;
  float _normalizationGain;
  std::vector<Real> _frameHistory;
  std::vector<Real> _tmpFrame;

  bool _configured;

 public:
  OverlapAdd() {
    declareInput(_frames, "frame", "the windowed input audio frame");
    declareOutput(_output, "signal", "the output overlap-add audio signal");
    _output.setBufferType(BufferUsage::forLargeAudioStream);
  }
  ~OverlapAdd() {}

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing the overlap-add process", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the overlap-add function is computed", "(0,inf)", 128);
    declareParameter("gain", "the normalization gain that scales the output signal. Useful for IFFT output", "(0.,inf)", 1.);
  }

  void reset();
  void configure();
  AlgorithmStatus process();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_OVERLAPADD_H
