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

#ifndef ESSENTIA_LOGSPECTRUM_H
#define ESSENTIA_LOGSPECTRUM_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class LogSpectrum : public Algorithm {
 public:

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<std::vector<Real> > _logFreqSpectrum;
  Output<std::vector<Real> > _meanTuning;
  Output<Real> _localTuning;

 public:
  LogSpectrum() {
    declareInput(_spectrum, "spectrum", "spectrum frame");
    declareOutput(_logFreqSpectrum, "logFreqSpectrum", "log frequency spectrum frame");
    declareOutput(_meanTuning, "meanTuning", "normalized mean tuning frequency");
    declareOutput(_localTuning, "localTuning", "normalized local tuning frequency");
  }

  void declareParameters() {
    declareParameter("frameSize", "the input frame size of the spectrum vector", "(1,inf)", 1025);
    declareParameter("sampleRate", "the input sample rate", "(0,inf)", 44100.);
    declareParameter("rollOn", "this removes low-frequency noise - useful in quiet recordings", "[0,5]", 0.f);
    declareParameter("binsPerSemitone", " bins per semitone", "(0,inf)", 3.0);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
  static const Real precision;

 protected:
  int _frameCount;
  int _nBPS;
  int _nOctave;
  int _nNote;
  size_t _frameSize;
  Real _sampleRate;
  Real _rollon;
  std::vector<int> _kernelFftIndex;
  std::vector<int> _kernelNoteIndex;
  std::vector<Real> _meanTunings;
  std::vector<Real> _localTunings;
  std::vector<Real> _kernelValue;
  std::vector<Real> _sinvalues;
  std::vector<Real> _cosvalues;

  bool logFreqMatrix(Real fs, int frameSize, std::vector<Real> &outmatrix);
  Real cospuls(Real x, Real centre, Real width);
  Real pitchCospuls(Real x, Real centre, int binsperoctave);
  void initialize();
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LogSpectrum : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _logFreqSpectrum;
  Source<std::vector<Real> > _meanTuning;
  Source<Real> _localTuning;

 public:
  LogSpectrum() {
    declareAlgorithm("LogSpectrum");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_logFreqSpectrum, TOKEN, "logFreqSpectrum");
    declareOutput(_meanTuning, TOKEN, "meanTuning");
    declareOutput(_localTuning, TOKEN, "localTuning");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_LOGSPECTRUM_H
