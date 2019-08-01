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
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _chromagram;

  Algorithm* _spectrumCQ;
  
  std::vector<Real> _chromaBuffer;

  unsigned _binsPerOctave;
  unsigned _octaves;

  enum NormalizeType {
        NormalizeNone,
        NormalizeUnitSum,
        NormalizeUnitMax
      }; 

  NormalizeType _normalizeType;
 
 public:
  Chromagram() {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_chromagram, "chromagram", "the magnitude constant-Q chromagram");

    _spectrumCQ = AlgorithmFactory::create("SpectrumCQ");
  }

  ~Chromagram() {
    delete _spectrumCQ;
  }

  void declareParameters() {
    declareParameter("minFrequency", "minimum frequency [Hz]", "[1,inf)", 32.7);
    declareParameter("numberBins", "number of frequency bins, starting at minFrequency", "[1,inf)", 84);
    declareParameter("binsPerOctave", "number of bins per octave", "[1,inf)", 12);
    declareParameter("sampleRate", "FFT sampling rate [Hz]", "[0,inf)", 44100.);
    declareParameter("threshold", "bins whose magnitude is below this quantile are discarded", "[0,1)", 0.01);
    declareParameter("scale", "filters scale. Larger values use longer windows", "[0,inf)", 1.0);
    declareParameter("windowType", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("minimumKernelSize", "minimum size allowed for frequency kernels", "[2,inf)", 4);
    declareParameter("normalizeType", "normalize type", "{none,unit_sum,unit_max}", "unit_max");
    declareParameter("zeroPhase", "a boolean value that enables zero-phase windowing. Input audio frames should be windowed with the same phase mode", "{true,false}", true);
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
  Sink<std::vector<Real> > _signal;
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
