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

#ifndef ESSENTIA_MULTIPLEXER_H
#define ESSENTIA_MULTIPLEXER_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class Multiplexer : public Algorithm {
 protected:
  std::vector<Sink<Real>*> _realInputs;
  std::vector<Sink<std::vector<Real> >*> _vectorRealInputs;

  Source<std::vector<Real> > _output;

  void clearInputs();

 public:

  Multiplexer() : Algorithm() {
    declareOutput(_output, 1, "data", "the frame containing the input values and/or input frames");
  }


  ~Multiplexer() {
    clearInputs();
  }

  void declareParameters() {
    declareParameter("numberRealInputs", "the number of inputs of type Real to multiplex", "[0,inf)", 0);
    declareParameter("numberVectorRealInputs", "the number of inputs of type vector<Real> to multiplex", "[0,inf)", 0);
  }

  void configure();

  // inputs should be named real_0, real_1, ..., vector_0, vector_1, ...
  SinkBase& input(const std::string& name);


  AlgorithmStatus process();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#include "algorithm.h"

namespace essentia {
namespace standard {

// The std version could have used the streaming multiplexer, but did not
// implemented this way, cause I find simpler not having to use vectorInput and vectoOutput
class Multiplexer : public Algorithm {
 protected:
   std::vector<Input<std::vector<Real> >* > _realInputs;
   std::vector<Input<std::vector<std::vector<Real> > >* >_vectorRealInputs;
   Output<std::vector<std::vector<Real> > > _output;

  void clearInputs();

 public:
  Multiplexer() {
    declareOutput(_output, "data", "the frame containing the input values and/or input frames");
  }

  ~Multiplexer() {
    clearInputs();
  }

  void declareParameters() {
    declareParameter("numberRealInputs", "the number of inputs of type Real to multiplex", "[0,inf)", 0);
    declareParameter("numberVectorRealInputs", "the number of inputs of type vector<Real> to multiplex", "[0,inf)", 0);
  }

  void configure();

  // inputs should be named real_0, real_1, ..., vector_0, vector_1, ...
  InputBase& input(const std::string& name);

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_MULTIPLEXER_H
