/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWINPUTTEMPOCNN_H
#define ESSENTIA_TENSORFLOWINPUTTEMPOCNN_H

#include "essentiamath.h"
#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class TensorflowInputTempoCNN : public Algorithm {

 protected:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _bands;

  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _melBands;

 public:
  TensorflowInputTempoCNN() {
    declareInput(_frame, "frame", "the audio frame");
    declareOutput(_bands, "bands", "the mel bands");

    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _melBands = AlgorithmFactory::create("MelBands");
  }

  ~TensorflowInputTempoCNN() {
    if (_windowing) delete _windowing;
    if (_spectrum) delete _spectrum;
    if (_melBands) delete _melBands;
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
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TensorflowInputTempoCNN : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _bands;

 public:
  TensorflowInputTempoCNN() {
    declareAlgorithm("TensorflowInputTempoCNN");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_bands, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TENSORFLOWINPUTTEMPOCNN_H
