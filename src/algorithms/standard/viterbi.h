/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_VITERBI_H
#define ESSENTIA_VITERBI_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Viterbi : public Algorithm {

 protected:
  Input<std::vector<std::vector<Real> > > _observationProbabilities;
  Input<std::vector<Real> > _initialization;
  Input<std::vector<size_t> > _fromIndex;
  Input<std::vector<size_t> > _toIndex;
  Input<std::vector<Real> > _transitionProbabilities;
  Output<std::vector<int> > _path;

  std::vector<int> _tempPath; 

 public:
  Viterbi() {
    declareInput(_observationProbabilities, "observationProbabilities", "the observation probabilities");
    declareInput(_initialization, "initialization", "the initialization");
    declareInput(_fromIndex, "fromIndex", "the transition matrix from index");
    declareInput(_toIndex, "toIndex", "the transition matrix to index");
    declareInput(_transitionProbabilities, "transitionProbabilities", "the transition probabilities matrix");
    declareOutput(_path, "path", "the decoded path");
  }

  ~Viterbi() {
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Viterbi : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _observationProbabilities;
  Sink<std::vector<Real> > _initialization;
  Sink<std::vector<size_t> > _fromIndex;
  Sink<std::vector<size_t> > _toIndex;
  Sink<std::vector<Real> > _transitionProbabilities;
  Source<std::vector<int> > _path;

 public:
  Viterbi() {
    declareAlgorithm("Viterbi");
    declareInput(_observationProbabilities, TOKEN, "observationProbabilities");
    declareInput(_initialization, TOKEN, "initialization");
    declareInput(_fromIndex , TOKEN, "fromIndex");
    declareInput(_toIndex, TOKEN, "toIndex");
    declareInput(_transitionProbabilities, TOKEN, "transitionProbabilities");
    declareOutput(_path, TOKEN, "path");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLATNESS_H
