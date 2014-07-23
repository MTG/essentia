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
  Output<std::vector<Real> > _diffs;


  
  	int _binW;
  	int _frameWi;
	bool _online;


	Algorithm* _maxf;


 public:
  SuperFluxNovelty() {
    declareInput(_bands, "bands", "the input bands spectrogram");
    declareOutput(_diffs, "Differences", "SuperFluxNoveltyd input");
	_maxf = AlgorithmFactory::create("MaxFilter");
    
  }

  ~SuperFluxNovelty() {

  }

  void declareParameters() {
    declareParameter("binWidth", "height(n of frequency bins) of the SuperFluxNoveltyFilter", "[3,inf)", 3);
	declareParameter("frameWidth", "differenciate with the N-th previous frame", "(0,inf)", 2);
	declareParameter("Online", "realign output with audio by frameWidth ; if using streaming mode : set it to true, else for static precision measurement: use false", "{false,true}", false);
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

class SuperFluxNovelty : public Algorithm {

 protected:
  Sink< vector<Real> > _bands;
   Source<Real  > _diffs;
  
  
  essentia::standard::Algorithm* _algo;

int bufferSize=10;

 public:
  SuperFluxNovelty(){
    _algo = standard::AlgorithmFactory::create("SuperFluxNovelty");
    declareInput(_bands, bufferSize,1,"bands","the input bands spectrogram");
    declareOutput(_diffs,1,"Differences","SuperFlux");

  }




  void declareParameters() {
    declareParameter("binWidth", "height(n of frequency bins) of the SuperFluxNoveltyFilter", "[3,inf)", 3);
	declareParameter("frameWidth", "number of frame for differentiation", "(0,inf)", 2);
	declareParameter("Online", "realign output with audio by frameWidth : if using streaming mode set it to true, else for static precision measurement, use false", "{false,true}", true);
  }
   
void configure() {
     _algo->configure(_params);
	   _bands.setAcquireSize(_algo->parameter("frameWidth").toInt()+1);
     _bands.setReleaseSize(1);

   // if(!_algo->parameter("Online").toBool()) throw EssentiaException("atm, can not be in streaming mode without online option set to true");

  }

  AlgorithmStatus process();
  void reset(){
//   _algo->reset();
  };

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SuperFluxNovelty_H
