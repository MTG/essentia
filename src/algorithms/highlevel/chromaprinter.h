/*
 * Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_CHROMAPRINTER_H
#define ESSENTIA_CHROMAPRINTER_H

#include "algorithmfactory.h"
#include <chromaprint.h>
#include "essentiamath.h"

namespace essentia {
namespace standard {

class Chromaprinter : public Algorithm {

 protected:
  Input<std::vector<Real> >  _signal;
  Output<std::string> _fingerprint;

  Real _sampleRate;
  Real _maxLength;
  ChromaprintContext *_ctx;

 public:
  Chromaprinter() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_fingerprint, "fingerprint", "the chromaprint value");
  }

  ~Chromaprinter() {}

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("maxLength", "maximum duration of the chromaprint. 0 to use the full audio length [s]", "[0,inf)", 0.);
  }

  void reset() {}

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

//#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

/*class Chromaprinter : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _fingerprint;

 public:
  Chromaprinter() {
    declareAlgorithm("ChromaprintGenerator");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_fingerprint, TOKEN, "fingerprint");
  }
};*/

class Chromaprinter : public Algorithm {
 protected:
  Sink<Real> _signal;
  Source<std::string> _fingerprint;

  Real _sampleRate;
  Real _analysisTime;

  int _inputSize;

  ChromaprintContext *_ctx;

 public:
  Chromaprinter() : Algorithm() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_fingerprint, "fingerprint", "the chromaprint value");

    _fingerprint.setBufferType(BufferUsage::forMultipleFrames);
  }

  ~Chromaprinter() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("analysisTime", "A chromaprint is retrieved each analysisTime seconds. 0 to use the full audio length [s]", "(0,inf)", 5.);
  }

  void configure() {
    _sampleRate = parameter("sampleRate").toReal();
    _analysisTime = parameter("analysisTime").toReal();
    _inputSize = _sampleRate * _analysisTime;

    _signal.setAcquireSize(_inputSize);
    _signal.setReleaseSize(_inputSize);

    _fingerprint.setAcquireSize(1);
    _fingerprint.setReleaseSize(1);
  }

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CHROMAPRINTER_H

