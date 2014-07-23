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

#ifndef ESSENTIA_SuperFluxPeaks_H
#define ESSENTIA_SuperFluxPeaks_H

#include "algorithmfactory.h"
using namespace std;
namespace essentia {
namespace standard {

class SuperFluxPeaks : public Algorithm {

 private:
 	Input<std::vector<Real> > _signal;
 	Output<std::vector<Real> > _peaks;
 
	standard::Algorithm* _movAvg;
	standard::Algorithm* _maxf;
  
 	int _pre_avg;
	int _pre_max;
	Real _combine;
// 	Real _threshold;


  	int hopSize;
  	Real frameRate;

	bool _rawMode;
	bool _startZero;

int lastPidx ;

 public:
  SuperFluxPeaks() {
    declareInput(_signal, "novelty", "the input novelty");
	declareOutput(_peaks, "peaks", "the input novelty");
    _movAvg = AlgorithmFactory::create("MovingAverage");
    _maxf = AlgorithmFactory::create("MaxFilter");
    
  }




  void declareParameters() {
	declareParameter("frameRate", "frameRate", "(0,inf)", 172.);
    declareParameter("threshold", "threshold for peak-picking", "(0,inf)", 1.25);
	declareParameter("combine", "ms for onset combination", "(0,inf)", 30);
    declareParameter("pre_avg", "use N miliseconds past information for moving average", "(0,inf)", 100);
	declareParameter("pre_max", "use N miliseconds past information for moving maximum", "(0,inf)", 30);
	declareParameter("rawmode", "output mode: if true, returns array of same size as novelty function, with 1 where peaks stands, if false, output list of peaks instants", "{true,false}", false);
	declareParameter("startFromZero", "in rawmode, output starts at 0 if not starts at frame corresponding max(pre_avg,pre_max)", "{true,false}", true);

}

  void reset() {Algorithm::reset();};
  void configure();
  void compute();


  static const char* name;
  static const char* version;
  static const char* description;
};

} // namespace standard
} // namespace essentia




namespace essentia {
namespace streaming {

class SuperFluxPeaks : public Algorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _peaks;

  standard::Algorithm * _algo;

bool _rawmode;

 public:
  SuperFluxPeaks(){
  
    _algo = standard::AlgorithmFactory::create("SuperFluxPeaks");
    declareInput(_signal, 17,1,"novelty","the input bands spectrogram");
    declareOutput(_peaks,1,1,"peaks","SuperFlux");
  };
  

  void declareParameters() {
	declareParameter("frameRate", "frameRate", "(0,inf)", 172.);
    declareParameter("threshold", "threshold for peak-picking", "(0,inf)", 1.25);
	declareParameter("combine", "ms for onset combination", "(0,inf)", 30.0);
    declareParameter("pre_avg", "use N miliseconds past information for moving average", "(0,inf)", 100);
	declareParameter("pre_max", "use N miliseconds past information for moving maximum", "(0,inf)", 30);
	declareParameter("rawmode", "output mode: if true, returns array of same size as novelty function, with 1 where peaks stands, if false, output list of peaks instants", "{true,false}", true);
	declareParameter("startFromZero", "if true; output starts at 0, if false; starts at frame corresponding max(pre_avg,pre_max)", "{true,false}", false);
  };


  // link algo parameter with streaming burffer options

void configure(){
//EXEC_DEBUG("configuring Peaks");
_algo->configure(this->_params);
int aqS =  _algo->parameter("frameRate").toReal() * max(_algo->parameter("pre_avg").toInt(),_algo->parameter("pre_max").toInt()) / 1000;
    EXEC_DEBUG("setAcquireSize" << aqS);
    _signal.setAcquireSize(aqS);
    _signal.setReleaseSize(1);
    
    
    if(_algo->parameter("rawmode").toBool() && !_algo->parameter("startFromZero").toBool()){
    _peaks.setAcquireSize(1);
    _peaks.setReleaseSize(1);
    }
    else{
      cout<<"aq"<<aqS<<endl;
    _peaks.setAcquireSize(aqS);
    _peaks.setReleaseSize(aqS);
    }


};




  AlgorithmStatus process();


  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SuperFluxPeaks_H
