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

#ifndef ESSENTIA_SCALE_H
#define ESSENTIA_SCALE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Scale : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _scaled;

  Real _factor, _maxValue;
  bool _clipping;

 public:
  Scale() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_scaled, "signal", "the output audio signal");
  }

  void declareParameters() {
    declareParameter("factor", "the multiplication factor by which the audio will be scaled", "[0,inf)", 10.0);
    declareParameter("clipping", "boolean flag whether to apply clipping or not", "{true,false}", true);
    declareParameter("maxAbsValue", "the maximum value above which to apply clipping", "[0,inf)", 1.0);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Scale : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _scaled;

 public:
  Scale() {
    int preferredSize = 4096;
    declareAlgorithm("Scale");
    declareInput(_signal, STREAM, preferredSize, "signal");
    declareOutput(_scaled, STREAM, preferredSize, "signal");

    _scaled.setBufferType(BufferUsage::forLargeAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SCALE_H
