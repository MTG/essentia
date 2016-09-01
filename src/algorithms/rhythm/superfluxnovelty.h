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

#ifndef ESSENTIA_SUPERFLUXNOVELTY_H
#define ESSENTIA_SUPERFLUXNOVELTY_H

#include "algorithmfactory.h"
using namespace std;

namespace essentia {
namespace standard {

class SuperFluxNovelty : public Algorithm {

 private:
  Input<std::vector< std::vector<Real> >  > _bands;
  Output<Real> _diffs;

 	int _binWidth;
  int _frameWidth;

  Algorithm* _maxFilter;

 public:
  SuperFluxNovelty() {
    declareInput(_bands, "bands", "the input bands spectrogram");
    declareOutput(_diffs, "differences", "SuperFlux novelty curve");
	 _maxFilter = AlgorithmFactory::create("MaxFilter"); 
  }

  ~SuperFluxNovelty() {
    delete _maxFilter;
  }

  void declareParameters() {
    declareParameter("binWidth", "filter width (number of frequency bins)", "[3,inf)", 3);
    declareParameter("frameWidth", "differentiation offset (compute the difference with the N-th previous frame)", "(0,inf)", 2);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* category;  
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class SuperFluxNovelty : public Algorithm {

 protected:
  Sink< vector<Real> > _bands;
  Source<Real  > _diffs;

  essentia::standard::Algorithm* _algo;

 public:
  SuperFluxNovelty() {
    declareInput(_bands, "bands", "the input bands spectrogram");
    declareOutput(_diffs,1,1, "differences", "SuperFlux novelty curve");
    _algo = essentia::standard::AlgorithmFactory::create("SuperFluxNovelty");
  }
  
  ~SuperFluxNovelty() {
    delete _algo;
  }

  void declareParameters() {
    declareParameter("binWidth", "filter width (number of frequency bins)", "[3,inf)", 3);
    declareParameter("frameWidth", "differentiation offset (compute the difference with the N-th previous frame)", "(0,inf)", 2);
  }
   
  void configure() {
    _algo->configure(_params);
    _bands.setAcquireSize(_algo->parameter("frameWidth").toInt() + 1);
    _bands.setReleaseSize(1);
  }

  AlgorithmStatus process();
  void reset() {};

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SUPERFLUXNOVELTY_H
