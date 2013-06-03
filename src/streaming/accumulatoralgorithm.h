/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ACCUMULATORALGORITHM_H
#define ESSENTIA_ACCUMULATORALGORITHM_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

/**
 * An AccumulatorAlgorithm is a special class of streaming algorithm that behaves
 * in the following way:
 *  - during most of the processing, it just consumes whatever comes in on its
 *    defined input (unique), and use that to maintain an internal state
 *  - when the end of the stream is reached, it computes a resulting value and
 *    outputs it on its defined source(s).
 *
 * By subclassing the AccumulatorAlgorithm class, you get all the buffering and
 * data management done for you. In exchange, you just need to implement 2 methods:
 *
 *  - void consume();
 *      which will be called during the processing. When called, you can safely
 *      assume that the data has already been acquired on your defined input sink
 *
 *  - void finalProduce();
 *      which will be called at the end of the stream, when all the tokens will
 *      have been fed to the consume() method. You will have to explicitly push your
 *      result
 *
 *
 * You will also need to declare your input sink and output source using the
 * specialized @c declareInputStream and @c declareOutputResult methods, instead of
 * the standard @c declareInput and @c declareOutput ones.
 *
 * As an example, please refer to the source code of the TCToTotal algorithm.
 *
 * WARNING: declaring multiple input streams will result in undefined behavior.
 *          Multiple output results are fine, though.
 *
 * WARNING: if you overload the reset() method, do not forget to call the base class
 *          implementation in it.
 */
class AccumulatorAlgorithm : public Algorithm {
 public:
  AccumulatorAlgorithm();

  AlgorithmStatus process();

  void reset();

  virtual void consume() = 0;
  virtual void finalProduce() = 0;

  void declareInputStream(SinkBase& sink, const std::string& name, const std::string& desc,
                          int preferredAcquireSize = 4096);

  void declareOutputResult(SourceBase& source, const std::string& name, const std::string& desc);

 protected:
  int _preferredSize;
  SinkBase* _inputStream;

  // shadow those methods to prevent people from using them
  void declareInput();
  void declareOutput();
};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ACCUMULATORALGORITHM_H
