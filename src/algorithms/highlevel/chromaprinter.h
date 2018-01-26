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
    declareOutput(_fingerprint, "fingerprint", "the chromaprint as a base64-encoded string");
  }

  ~Chromaprinter() {}

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("maxLength", "use the first 'maxLength' seconds to compute the chromaprint. 0 to use the full audio length [s]", "[0,inf)", 0.);
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

  std::vector<int16_t> _buffer;

  unsigned  _chromaprintSize;
  unsigned _count;


  ChromaprintContext *_ctx;

  bool _ok;
  bool _returnChromaprint;
  bool _concatenate;

  std::string getChromaprint();
  void initChromaprint();

  std::string fingerprintConcatenated;

 public:
  Chromaprinter() : Algorithm() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_fingerprint, "fingerprint", "the chromaprint as a base64-encoded string");

    _fingerprint.setBufferType(BufferUsage::forMultipleFrames);
  }

  ~Chromaprinter() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("analysisTime", "a chromaprint is computed each 'analysisTime' seconds. It is not recommended use a value lower than 30.", "(0,inf)", 30.);
    declareParameter("concatenate", "if true, chromaprints are concatenated and returned as a single string. Otherwise a chromaprint is returned each 'analysisTime' seconds.", "{true,false}", true);
  }

  void configure();



  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CHROMAPRINTER_H

