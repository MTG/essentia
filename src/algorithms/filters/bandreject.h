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

#ifndef ESSENTIA_BANDREJECT_H
#define ESSENTIA_BANDREJECT_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {


class BandReject : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  Algorithm* _filter;

 public:
  BandReject() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~BandReject() {
    if(_filter) delete _filter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("cutoffFrequency", "the cutoff frequency for the filter [Hz]", "(0,inf)", 1500.);
    declareParameter("bandwidth", "the bandwidth of the filter [Hz]", "(0,inf)", 500.);
  }

  void reset() {
    _filter->reset();
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};


} // namespace standard
namespace streaming {


class BandReject : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  BandReject() {
    declareAlgorithm("BandReject");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BANDREJECT_H
