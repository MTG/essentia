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

#ifndef ESSENTIA_SUPERFLUXNOVELTY_H
#define ESSENTIA_SUPERFLUXNOVELTY_H

#include "algorithmfactory.h"
using namespace std;

namespace essentia {
namespace standard {

class SuperFluxNovelty : public Algorithm{

 private:
 
  Input<std::vector< std::vector<Real> >  > _bands;
  Output<Real > _diffs;


  
  	int _binW;
  	int _frameWi;
	


	Algorithm* _maxf;


 public:
  SuperFluxNovelty() {
    declareInput(_bands, "bands", "the input bands spectrogram");
    declareOutput(_diffs, "Differences", "SuperFluxNovelty input");
	_maxf = AlgorithmFactory::create("MaxFilter");
    
  }

  ~SuperFluxNovelty() {
      delete _maxf;
  }

  void declareParameters() {
    declareParameter("binWidth", "height(n of frequency bins) of the SuperFluxNoveltyFilter", "[3,inf)", 3);
	declareParameter("frameWidth", "differenciate with the N-th previous frame", "(0,inf)", 2);
	
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
// #include "streamingalgorithmcomposite.h"
namespace essentia {
namespace streaming {

class SuperFluxNovelty : public Algorithm
    {

 protected:
    Sink< vector<Real> > _bands;
    Source<Real  > _diffs;
  
  
    essentia::standard::Algorithm* _algo;


 public:

        SuperFluxNovelty(){
            declareInput(_bands, "bands", "the input bands spectrogram");
            declareOutput(_diffs,1,1, "Differences", "SuperFluxNovelty input");
            _algo = essentia::standard::AlgorithmFactory::create("SuperFluxNovelty");

        }
        ~SuperFluxNovelty(){
            delete _algo;
        }


  void declareParameters() {
    declareParameter("binWidth", "height(n of frequency bins) of the SuperFluxNoveltyFilter", "[3,inf)", 3);
	declareParameter("frameWidth", "number of frame for differentiation", "(0,inf)", 2);
    }
   
void configure() {
     _algo->configure(_params);
    _bands.setAcquireSize(_algo->parameter("frameWidth").toInt()+1);
    _bands.setReleaseSize(1);

  }

  AlgorithmStatus process();
  void reset(){
  };

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SuperFluxNovelty_H
