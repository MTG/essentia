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

#ifndef ESSENTIA_CHROMAGRAM_H
#define ESSENTIA_CHROMAGRAM_H

#include "algorithmfactory.h"
#include <complex>
#include "constantq.h"


namespace essentia {
namespace standard {

class Chromagram : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > > _signal;
  Output<std::vector<Real> > _chromagram;

  Algorithm* _constantq;
  Algorithm* _magnitude;
  
  std::vector<std::complex<Real> > _CQBuffer;
  std::vector<Real> _ChromaBuffer;

  std::vector<double> _CQdata;
  unsigned int _binsPerOctave;
  unsigned _octaves;

  enum NormalizeType {
        NormalizeNone,
        NormalizeUnitSum,
        NormalizeUnitMax
      }; 

  NormalizeType _normalizeType;
 
 public:
  Chromagram() {
    declareInput(_signal, "frame", "the input frame (complex)");
    declareOutput(_chromagram, "chromagram", "the magnitude chromagram of the input audio signal");

    _constantq = AlgorithmFactory::create("ConstantQ");
    _magnitude = AlgorithmFactory::create("Magnitude"); 
  }

  ~Chromagram() {
    delete _constantq;
    delete _magnitude;
  }

  void declareParameters() {
    declareParameter("minFrequency", "minimum frequency [Hz]", "[1,inf)", 32.7);
    declareParameter("numberBins", "number of frequency bins, starting at minFrequency", "[1,inf)", 84);
    declareParameter("binsPerOctave", "number of bins per octave", "[1,inf)", 12);    
    declareParameter("sampleRate", "FFT sampling rate [Hz]", "[0,inf)", 44100.);  
    declareParameter("threshold", "threshold value", "[0,inf)", 0.0005);
    // TODO: explain threshold better 
    declareParameter("normalizeType", "normalize type", "{none,unit_sum,unit_max}", "unit_max");   
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Chromagram : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _signal;
  Source<std::vector<Real> > _chromagram;

 public:
  Chromagram() {
    declareAlgorithm("Chromagram");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_chromagram, TOKEN, "chromagram");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CHROMAGRAM_H
