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

#ifndef ESSENTIA_EQUALLOUDNESS_H
#define ESSENTIA_EQUALLOUDNESS_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class EqualLoudness : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  std::vector<Real> _z; // intermediate storage vector

  Algorithm* _yulewalkFilter;
  Algorithm* _butterworthFilter;

 public:
  EqualLoudness() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");

    _yulewalkFilter = AlgorithmFactory::create("IIR");
    _butterworthFilter = AlgorithmFactory::create("IIR");
  }

  ~EqualLoudness() {
    delete _yulewalkFilter;
    delete _butterworthFilter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "{32000,44100,48000}", 44100.);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class EqualLoudness : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  EqualLoudness() {
    declareAlgorithm("EqualLoudness");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");

    _y.setBufferType(BufferUsage::forLargeAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_EQUALLOUDNESS_H
