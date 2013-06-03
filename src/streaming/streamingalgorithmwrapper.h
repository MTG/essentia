/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_STREAMINGALGORITHMWRAPPER_H
#define ESSENTIA_STREAMINGALGORITHMWRAPPER_H

#include "streamingalgorithm.h"
#include "algorithm.h"

namespace essentia {
namespace streaming {

enum NumeralType {
  TOKEN,
  STREAM
};


class StreamingAlgorithmWrapper : public Algorithm {

 protected:
  typedef EssentiaMap<std::string, NumeralType> NumeralTypeMap;

  NumeralTypeMap _inputType, _outputType; // indicates whether the algo takes a single token or a sequence of tokens
  standard::Algorithm* _algorithm;
  int _streamSize;

 public:

  StreamingAlgorithmWrapper() : _algorithm(0) {}
  ~StreamingAlgorithmWrapper();

  void declareInput(SinkBase& sink, NumeralType type, const std::string& name);
  void declareInput(SinkBase& sink, NumeralType type, int n, const std::string& name);

  void declareOutput(SourceBase& source, NumeralType type, const std::string& name);
  void declareOutput(SourceBase& source, NumeralType type, int n, const std::string& name);


  void synchronizeInput(const std::string& name);
  //void synchronizeInput(SinkBase* input);
  void synchronizeOutput(const std::string& name);
  //void synchronizeOutput(SourceBase* output);

  void synchronizeIO();

  void declareAlgorithm(const std::string& name);

  void configure(const ParameterMap& params) {
    _algorithm->configure(params);
    this->setParameters(params);
  }

  void configure() {
    _algorithm->configure();
  }

  void reset() {
    Algorithm::reset();
    E_DEBUG(EAlgorithm, "Standard : " << name() << "::reset()");
    _algorithm->reset();
    E_DEBUG(EAlgorithm, "Standard : " << name() << "::reset() ok!");
  }

  void setParameters(const ParameterMap& params) {
    Configurable::setParameters(params);
    _algorithm->setParameters(params);
  }

  void declareParameters() {
    _algorithm->declareParameters();
    _params = _defaultParams = _algorithm->defaultParameters();
    parameterRange = _algorithm->parameterRange;
    parameterDescription = _algorithm->parameterDescription;
  }

  AlgorithmStatus process();

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMINGALGORITHMWRAPPER_H
