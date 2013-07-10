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

#ifndef ESSENTIA_DCREMOVAL_H
#define ESSENTIA_DCREMOVAL_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class DCRemoval : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _signalDC;

  Algorithm* _filter;

 public:
  DCRemoval() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_signalDC, "signal", "the filtered signal, with the DC component removed");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~DCRemoval() {
    delete _filter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("cutoffFrequency", "the cutoff frequency for the filter [Hz]", "(0,inf)", 40.);
  }

  void reset() {
    _filter->reset();
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class DCRemoval : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _signalDC;

  static const int preferredSize = 4096;

 public:
  DCRemoval() {
    declareAlgorithm("DCRemoval");
    declareInput(_signal, STREAM, preferredSize, "signal");
    declareOutput(_signalDC, STREAM, preferredSize, "signal");

    _signalDC.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DCREMOVAL_H
