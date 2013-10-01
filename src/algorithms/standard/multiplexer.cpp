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

#include "multiplexer.h"
using namespace std;

namespace essentia {
namespace streaming {

const char* Multiplexer::name = "Multiplexer";
const char* Multiplexer::description = DOC("This algorithm returns a single vector from a given number of real values and/or frames. Frames from different inputs are multiplexed onto a single stream in an alternating fashion.\n"
"\n"
"This algorithm throws an exception if the number of input reals (or vector<real>) is less than the number specified in configuration parameters or if the user tries to acces an input which has not been specified.\n"
"\n"
"References:\n"
"  [1] Multiplexer - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Multiplexer\n");


void Multiplexer::clearInputs() {
  for (int i=0; i<int(_realInputs.size()); i++) delete _realInputs[i];
  for (int i=0; i<int(_vectorRealInputs.size()); i++) delete _vectorRealInputs[i];
  _realInputs.clear();
  _vectorRealInputs.clear();
  _inputs.clear(); // needed to maintain input order
}


void Multiplexer::configure() {
  clearInputs();

  int num = parameter("numberRealInputs").toInt();
  for (int i=0; i<num; i++) {
    _realInputs.push_back(new Sink<Real>());
    ostringstream inputName; inputName << "real_" << i;
    ostringstream inputIndex; inputIndex << i;
    declareInput(*_realInputs.back(), 1, inputName.str(), "signal input #" + inputIndex.str());
  }

  num = parameter("numberVectorRealInputs").toInt();
  for (int i=0; i<num; i++) {
    _vectorRealInputs.push_back(new Sink<vector<Real> >());
    ostringstream inputName; inputName << "vector_" << i;
    ostringstream inputIndex; inputIndex << i;
    declareInput(*_vectorRealInputs.back(), 1, inputName.str(), "frame input #" + inputIndex.str());
  }
}

// inputs should be named real_0, real_1, ..., vector_0, vector_1, ...
SinkBase& Multiplexer::input(const string& name) {
  if (name.substr(0, 5) == "real_") {
    istringstream parser(name.substr(5));
    int inputNumber;
    parser >> inputNumber;
    if (inputNumber > int(_realInputs.size())) {
      throw EssentiaException("Multiplexer: not enough real inputs: ", inputNumber);
    }
    return *_realInputs[inputNumber];
  }

  else if (name.substr(0, 7) == "vector_") {
    istringstream parser(name.substr(7));
    int inputNumber;
    parser >> inputNumber;
    if (inputNumber > int(_vectorRealInputs.size())) {
      throw EssentiaException("Multiplexer: not enough vector<real> inputs: ", inputNumber);
    }
    return *_vectorRealInputs[inputNumber];
  }

  else {
    throw EssentiaException("unknown input name: ", name);
  }
}

AlgorithmStatus Multiplexer::process() {
  EXEC_DEBUG("process()");

  AlgorithmStatus status = acquireData();
  if (status != OK) return status;

  EXEC_DEBUG("acquired successfully");

  vector<Real>& output = _output.firstToken();
  output.clear();

  for (int i=0; i<(int)_realInputs.size(); i++) {
    output.push_back(_realInputs[i]->firstToken());
  }

  for (int i=0; i<(int)_vectorRealInputs.size(); i++) {
    const vector<Real>& frame = _vectorRealInputs[i]->firstToken();
    for (int j=0; j<(int)frame.size(); j++) {
      output.push_back(frame[j]);
    }
  }

  EXEC_DEBUG("releasing data");
  releaseData();

  return OK;
}


} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* Multiplexer::name = "Multiplexer";
const char* Multiplexer::description = DOC("This algorithm returns a single vector from a given number of real values and/or frames. Frames from different inputs are multiplexed onto a single stream in an alternating fashion.\n"
"\n"
"This algorithm throws an exception if the number of input reals (or vector<real>) is less than the number specified in configuration parameters or if the user tries to acces an input which has not been specified.\n"
"\n"
"References:\n"
"  [1] Multiplexer - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Multiplexer\n");

void Multiplexer::clearInputs() {
  for (int i=0; i<int(_realInputs.size()); i++) delete _realInputs[i];
  for (int i=0; i<int(_vectorRealInputs.size()); i++) delete _vectorRealInputs[i];
  _realInputs.clear();
  _vectorRealInputs.clear();
  _inputs.clear(); // needed to maintain input order
}

void Multiplexer::configure() {
  int numReals = parameter("numberRealInputs").toInt();
  int numVectors = parameter("numberVectorRealInputs").toInt();
  for (int i=0; i<numReals; ++i) {
    _realInputs.push_back(new Input<vector<Real> >());
    ostringstream inputName; inputName << "real_" << i;
    ostringstream inputIndex; inputIndex << i;
    declareInput(*_realInputs.back(), inputName.str(), "signal input #" + inputIndex.str());
  }
  for (int i=0; i<numVectors; i++) {
    _vectorRealInputs.push_back(new Input<vector<vector<Real> > >());
    ostringstream inputName; inputName << "vector_" << i;
    ostringstream inputIndex; inputIndex << i;
    declareInput(*_vectorRealInputs.back(), inputName.str(), "frame input #" + inputIndex.str());
  }
}

InputBase& Multiplexer::input(const string& name) {
  if (name.substr(0, 5) == "real_") {
    istringstream parser(name.substr(5));
    int inputNumber;
    parser >> inputNumber;
    if (inputNumber > int(_realInputs.size())) {
      throw EssentiaException("Multiplexer: not enough real inputs: ", inputNumber);
    }
    return *_realInputs[inputNumber];
  }

  else if (name.substr(0, 7) == "vector_") {
    istringstream parser(name.substr(7));
    int inputNumber;
    parser >> inputNumber;
    if (inputNumber > int(_vectorRealInputs.size())) {
      throw EssentiaException("Multiplexer: not enough vector<real> inputs: ", inputNumber);
    }
    return *_vectorRealInputs[inputNumber];
  }

  else {
    throw EssentiaException("unknown input name: ", name);
  }
}

void Multiplexer::compute() {
    vector<vector<Real> >& output = _output.get();
    output.clear();
    int size = 0;
    int nFrames = 0;
    if (int(_realInputs.size())) nFrames = _realInputs[0]->get().size();
    else if (int(_vectorRealInputs.size())) nFrames = _vectorRealInputs[0]->get().size();
    else throw EssentiaException("Multiplexer: no inputs available");

    for (int i=0; i<int(_realInputs.size()); i++) {
      const vector<Real> vec = _realInputs[i]->get();
      if (int(vec.size()) != nFrames) {
        throw EssentiaException("Multiplexer: inputs with different length are not allowed");
      }
      //size += vec.size();
    }
    size = _realInputs.size();
    for (int i=0; i<int(_vectorRealInputs.size()); i++) {
      int maxVecSize = 0;
      const vector<vector<Real> >& frame = _vectorRealInputs[i]->get();
      if (int(frame.size()) != nFrames) {
        throw EssentiaException("Multiplexer: inputs with different length are not allowed");
      }
      for (int j=0; j<nFrames;++j) {
        if (int(frame[j].size()) > maxVecSize) maxVecSize = frame.size();
      }
      size += maxVecSize;
      //size += frame[0].size();
    }

    output.resize(nFrames); //size);

    for (int n=0; n<nFrames; n++) {
      output[n].reserve(size);
      for (int i=0; i<int(_realInputs.size()); i++) {
        output[n].push_back(_realInputs[i]->get()[n]);
      }

      for (int i=0; i<int(_vectorRealInputs.size()); i++) {
        const vector<Real>& frame = _vectorRealInputs[i]->get()[n];
        for (int j=0; j<int(frame.size()); j++) {
          output[n].push_back(frame[j]);
        }
      }
    }
}

} // namespace standard
} // namespace essentia
