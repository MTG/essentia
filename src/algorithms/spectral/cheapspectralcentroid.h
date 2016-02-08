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

#ifndef CHEAPSPECTRALCENTROID_H
#define CHEAPSPECTRALCENTROID_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class CheapSpectralCentroid : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _centroid;

  Real _sampleRate; /** sampling rate of the audio signal */

 public:
  CheapSpectralCentroid() {
    declareInput(_signal, "array", "the input array");
    declareOutput(_centroid, "centroid", "the spectral centroid of the signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "sampling rate of the input spectrum [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

}; //class CheapSpectralCentroid

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class CheapSpectralCentroid : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _centroid;

 public:
  CheapSpectralCentroid() {
    declareAlgorithm("CheapSpectralCentroid");
    declareInput(_signal, TOKEN, "array");
    declareOutput(_centroid, TOKEN, "centroid");
  }
};

} // namespace streaming
} // namespace essentia

#endif // CHEAPSPECTRALCENTROID_H
