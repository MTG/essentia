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

#ifndef ESSENTIA_BARKBANDS_H
#define ESSENTIA_BARKBANDS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class BarkBands : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

  Algorithm* _freqBands;

 public:
  BarkBands() {
    declareInput(_spectrumInput, "spectrum", "the input spectrum");
    declareOutput(_bandsOutput, "bands", "the energy of the bark bands");
    _freqBands = AlgorithmFactory::create("FrequencyBands");
  }

  ~BarkBands() {
    if (_freqBands) delete _freqBands;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "[0,inf)", 44100.);
    declareParameter("numberBands", "the number of desired barkbands", "[1,28]", 27);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BarkBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  BarkBands() {
    declareAlgorithm("BarkBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_BARKBANDS_H
