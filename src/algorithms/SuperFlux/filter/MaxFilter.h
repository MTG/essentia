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

#ifndef ESSENTIA_MAXFILTER_H
#define ESSENTIA_MAXFILTER_H

#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace standard {

class MaxFilter : public Algorithm {

 protected:
  Input< vector<Real>  > _array;
  Output< vector<Real> > _filtered;

  int _width;
  bool _causal;

 public:
  MaxFilter() {
    declareInput(_array, "signal", "signal to be filtered");
    declareOutput(_filtered, "signal", "filtered output ");
  }

  ~MaxFilter() {}

  void declareParameters() {
    declareParameter("width", "the window size, has to be odd if the window is centered", "[2,inf)", 3);
    declareParameter("causal", "use casual filter (window is behind current element otherwise it is centered around)", "{true,false}", true);
   // declareParameter("startFromZero", "suppress first frames width", "{true,false}", true); //TODO remove?
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class MaxFilter : public Algorithm {

 protected:
  Sink<Real > _array;
  Source<Real > _filtered;
 
  // TODO: remove? 
  //std::vector <Real> _buff;
  //int _idx;
 
 public:
  MaxFilter(){
    declareInput(_array, 1, 1, "signal", "signal to be filtered");
    declareOutput(_filtered, 1, 1, "signal","filtered output");
  }
  
  void declareParameters() {
    declareParameter("width", "window size for max filter : has to be odd as the window is centered on sample", "[3,inf)", 3);
  }

  void configure() {
    _array.setAcquireSize(parameter("width").toInt() + 1);
    _array.setReleaseSize(1);
	  //_buff.resize(parameter("width").toInt() + 1);
    // _filtered.setReleaseSize(1);
    //_idx =0; TODO remove?
  }
  
  AlgorithmStatus process();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MAXFILTER_H
