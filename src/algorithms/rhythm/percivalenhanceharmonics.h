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

#ifndef ESSENTIA_PERCIVALENHANCEHARMONICS_H
#define ESSENTIA_PERCIVALENHANCEHARMONICS_H

#include "algorithm.h"

namespace essentia {
namespace standard {
class PercivalEnhanceHarmonics : public Algorithm {

  protected:
    Input<std::vector<Real> > _input;
    Output<std::vector<Real> > _output;

  public:
    PercivalEnhanceHarmonics() {
    declareInput(_input, "array", "the input signal");
    declareOutput(_output, "array", "the input signal with enhanced harmonics");
    }
    ~PercivalEnhanceHarmonics(){
    }

    void declareParameters() {
    }

    void configure();
    void compute();
    void reset() {}

    static const char* name;
    static const char* category;
    static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PercivalEnhanceHarmonics : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input;
  Source<std::vector<Real> > _output;

 public:
  PercivalEnhanceHarmonics() {
    declareAlgorithm("PercivalEnhanceHarmonics");
    declareInput(_input, TOKEN, "array");
    declareOutput(_output, TOKEN, "array");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PERCIVALENHANCEHARMONICS_H
