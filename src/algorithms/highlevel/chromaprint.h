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

#ifndef ESSENTIA_CHROMAPRINT_H
#define ESSENTIA_CHROMAPRINT_H

#include "algorithmfactory.h"
#include "../../3rdparty/chromaprint-1.4.2/src/chromaprint.h"

namespace essentia {
namespace standard {

class Chromaprint : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::string> _fingerprint;

 public:
  Chromaprint() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_fingerprint, "fingerprint", "the chromaprint value");

  }

  ~Chromaprint() {}

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void reset() {}

  void configure(){};
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_CHROMAPRINT_H
