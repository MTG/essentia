/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_SINEMODELANAL_H
#define ESSENTIA_SINEMODELANAL_H

#include "algorithm.h"

// TODO: copy the file as part of Essentia. Check with Dmitry what is the best option
#include "../../../../musicbricks/sms-tools-master/software/models/utilFunctions_C/utilFunctions.h""


namespace essentia {
namespace standard {

class SineModelAnal : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _maxMagFreq;
  Real _sampleRate;

 public:
  SineModel() {
    declareInput(_spectrum, "spectrum", "the input spectrum (must have more than 1 element)");
    declareOutput(_freqs, "freqs", "the frequency with the largest magnitude [Hz]");
    declareOutput(_mags, "mags", "the frequency with the largest magnitude [Hz]");
      declareOutput(_phases, "phases", "the phases of the frequency with the largest magnitude [Hz]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure() {
    _sampleRate = parameter("sampleRate").toReal();
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SineModelAnal : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _maxMagFreq;

 public:
  SineModel() {
    declareAlgorithm("SineModelAnal");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_maxMagFreq, TOKEN, "maxMagFreq");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SINEMODELANAL_H
