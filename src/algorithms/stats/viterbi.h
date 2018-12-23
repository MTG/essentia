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
  Input<std::vector<std::vector<Real> > > _obs;
  Input<std::vector<Real> > _init;
  Input<std::vector<size_t> > _from;
  Input<std::vector<size_t> > _to;
  Input<std::vector<Real> > _transProb;
  Output<std::vector<int> > _path;

  std::vector<int> _tempPath; 

 public:
  Viterbi() {
    declareInput(_obs, "obs", "the observation probabilities");
    declareInput(_init, "init", "the initialization");
    declareInput(_from, "from", "the transition matrix from index");
    declareInput(_to, "to", "the transition matrix to index");
    declareInput(_transProb, "transProb", "the transition probablities matrix");
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
  Sink<std::vector<std::vector<Real> > > _obs;
  Sink<std::vector<Real> > _init;
  Sink<std::vector<size_t> > _from;
  Sink<std::vector<size_t> > _to;
  Sink<std::vector<Real> > _transProb;
  Source<std::vector<int> > _path;

 public:
  Viterbi() {
    declareAlgorithm("Viterbi");
    declareInput(_obs, TOKEN, "obs");
    declareInput(_init, TOKEN, "init");
    declareInput(_from , TOKEN, "from");
    declareInput(_to, TOKEN, "to");
    declareInput(_transProb, TOKEN, "transProb");
    declareOutput(_path, TOKEN, "path");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLATNESS_H
