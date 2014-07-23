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

#ifndef ESSENTIA_MaxFilter_H
#define ESSENTIA_MaxFilter_H

#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace standard {

class MaxFilter : public Algorithm {

 private:
 
  Input< vector<Real>  > _array;
  Output< vector<Real> > _filtered;


  
  	int _width;
  	bool _causal;

 public:
  MaxFilter() {
    declareInput(_array, "signal", "signal to be filtered");
    declareOutput(_filtered, "signal", "filtered output ");

    
  }

  ~MaxFilter() {

  }

  void declareParameters() {
    declareParameter("width", "window size for max filter :if centered, has to be odd ", "[2,inf)", 3);
    declareParameter("Causal", "if the filter is causal: windows is behind current element else windows is centered around ", "{true,false}", true);
   // declareParameter("startFromZero", "suppress first frames width", "{true,false}", true);

}

  void reset();
  void configure();
  void compute();


  static const char* name;
  static const char* version;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"
//#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class MaxFilter : public Algorithm {

 protected:
  Sink<Real > _array;
  Source<Real > _filtered;

 std::vector <Real> buff;
 int idx;
 

 public:
  MaxFilter(){
    declareInput(_array,1,1, "signal","signal");
    declareOutput(_filtered,1,1, "signal","signal");

  }
  
    void declareParameters() {
    declareParameter("width", "window size for max filter : has to be odd as the window is centered on sample", "[3,inf)", 3);
}
void configure() {
	_array.setAcquireSize(parameter("width").toInt()+1);
	_array.setReleaseSize(1);
	buff.resize(parameter("width").toInt()+1);
    // _filtered.setReleaseSize(1);
    idx =0;

  }
  
  AlgorithmStatus process();   
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MaxFilter_H
