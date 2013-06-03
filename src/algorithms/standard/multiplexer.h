/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
