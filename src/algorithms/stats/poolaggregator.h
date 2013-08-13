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

#ifndef ESSENTIA_POOLAGGREGATOR_H
#define ESSENTIA_POOLAGGREGATOR_H

#include <set>
#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class PoolAggregator : public Algorithm {

 protected:
  Input<Pool> _input;
  Output<Pool> _output;

  void aggregateSingleRealPool(const Pool& input, Pool& output);
  void aggregateRealPool(const Pool& input, Pool& output);
  void aggregateSingleVectorRealPool(const Pool& input, Pool& output);
  void aggregateVectorRealPool(const Pool& input, Pool& output);
  void aggregateArray2DRealPool(const Pool& input, Pool& output);
  void aggregateSingleStringPool(const Pool& input, Pool& output);
  void aggregateStringPool(const Pool& input, Pool& output);
  void aggregateVectorStringPool(const Pool& input, Pool& output);
  const std::vector<std::string>& getStats(const std::string& key) const;

  std::vector<std::string> _defaultStats;
  std::map<std::string, std::vector<std::string> > _exceptions;
  static const std::set<std::string> _supportedStats;

 public:
  PoolAggregator() {
    declareInput(_input, "input", "the input pool");
    declareOutput(_output, "output", "a pool containing the aggregate values of the input pool");
  }

  void declareParameters() {
    const char* defaultStatsC[] = { "mean", "var", "min", "max", "median" };
    std::vector<std::string> defaultStats = arrayToVector<std::string>(defaultStatsC);

    declareParameter("defaultStats", "the default statistics to be computed for each descriptor in the input pool", "", defaultStats);
    declareParameter("exceptions", "a mapping between descriptor names (no duplicates) and the types of statistics to be computed for those descriptors (e.g. { lowlevel.bpm : [min, max], lowlevel.gain : [var, min, dmean] })", "", std::map<std::string, std::vector<std::string> >());
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

// TODO: I'm not sure if the streaming is correct

namespace essentia {
namespace streaming {

class PoolAggregator : public StreamingAlgorithmWrapper {

 protected:
  Sink<Pool> _input;
  Source<Pool> _output;

 public:
  PoolAggregator() {
    declareAlgorithm("PoolAggregator");
    declareInput(_input, TOKEN, "input");
    declareOutput(_output, TOKEN, "output");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_POOLAGGREGATOR_H
