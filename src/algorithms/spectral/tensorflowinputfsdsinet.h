/*
 * Copyright (C) 2006-2023  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWINPUTFSDSINET_H
#define ESSENTIA_TENSORFLOWINPUTFSDSINET_H

#include "essentiamath.h"
#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class TensorflowInputFSDSINet : public Algorithm {

 protected:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _bands;

  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _melBands;
  Algorithm* _compression;

 public:
  TensorflowInputFSDSINet() {
    declareInput(_frame, "frame", "the audio frame");
    declareOutput(_bands, "bands", "the log-compressed mel bands");

    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _melBands = AlgorithmFactory::create("MelBands");
    _compression = AlgorithmFactory::create("UnaryOperator");
  }

  ~TensorflowInputFSDSINet() {
    if (_windowing) delete _windowing;
    if (_spectrum) delete _spectrum;
    if (_melBands) delete _melBands;
    if (_compression) delete _compression;
  }

  void declareParameters() {}

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:
  std::vector<Real> _windowedFrame;
  std::vector<Real> _spectrumFrame;
  std::vector<Real> _melBandsFrame;

  const int _fftSize = 2048;
  const int _frameSize = 660;
  const int _zeroPadding = _fftSize - _frameSize;
  const int _spectrumSize = _fftSize / 2 + 1;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TensorflowInputFSDSINet : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _bands;

 public:
  TensorflowInputFSDSINet() {
    declareAlgorithm("TensorflowInputFSDSINet");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_bands, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TENSORFLOWINPUTFSDSINET_H
