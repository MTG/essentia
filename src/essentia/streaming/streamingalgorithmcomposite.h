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

#ifndef ESSENTIA_STREAMINGALGORITHMCOMPOSITE_H
#define ESSENTIA_STREAMINGALGORITHMCOMPOSITE_H

#include "streamingalgorithm.h"
#include "sourceproxy.h"
#include "sinkproxy.h"

namespace essentia {
namespace streaming {


class ProcessStep {
protected:
  std::string _type;
  Algorithm* _algo;
public:
  ProcessStep(const std::string& type, Algorithm* algo) : _type(type), _algo(algo) {}
  const std::string& type() const { return _type; }
  Algorithm* algorithm() { return _algo; }
};

class ChainFrom : public ProcessStep {
public:
  ChainFrom(Algorithm* algo) : ProcessStep("chain", algo) {}
};

class SingleShot : public ProcessStep {
public:
  SingleShot(Algorithm* algo) : ProcessStep("single", algo) {}
};


class AlgorithmComposite : public Algorithm {

 public:

  // Those are available here because it doesn't make sense to require an
  // acquire/release size from a proxy source/sink
  void declareInput(SinkBase& sink, const std::string& name, const std::string& desc);
  void declareOutput(SourceBase& source, const std::string& name, const std::string& desc);

  // Those are here because otherwise they'd be shadowed
  void declareInput(SinkBase& sink, int n, const std::string& name, const std::string& desc) {
    Algorithm::declareInput(sink, n, name, desc);
  }
  void declareInput(SinkBase& sink, int acquireSize, int releaseSize, const std::string& name, const std::string& desc) {
    Algorithm::declareInput(sink, acquireSize, releaseSize, name, desc);
  }
  void declareOutput(SourceBase& source, int n, const std::string& name, const std::string& desc) {
    Algorithm::declareOutput(source, n, name, desc);
  }
  void declareOutput(SourceBase& source, int acquireSize, int releaseSize, const std::string& name, const std::string& desc) {
    Algorithm::declareOutput(source, acquireSize, releaseSize, name, desc);
  }


  /**
   * By default, does nothing, just waits for its inner algorithms to be
   * scheduled for execution by the task scheduler.
   */
  AlgorithmStatus process() { return PASS; }

  // needs to be declared so that the scheduler knows what to do with this
  virtual void declareProcessOrder() = 0;

  // TODO: can we make this return a const vector<ProcessStep>& instead?
  //  No: because we need to call the declareProcessOrder just at the moment we need
  //      to know it and not before, because the process order might be dependent on
  //      the configuration of the algorithm, e.g. see ReplayGain implementation
  std::vector<ProcessStep> processOrder();

  /**
   * Specialized implementation of the reset() method that will call reset() on all
   * the algorithms traversed by the defined process order.
   */
  void reset();

  void declareProcessStep(const ProcessStep& step);

protected:
  std::vector<ProcessStep> _processOrder;
};


} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMINGALGORITHMCOMPOSITE_H
