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

#ifndef ESSENTIA_NSGICONSTANTQ_H
#define ESSENTIA_NSGICONSTANTQ_H

#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class NSGIConstantQ : public Algorithm {
 protected:
  Output<std::vector<Real> > _signal;
  Input<std::vector<std::vector<std::complex<Real> > > >_constantQ ;
  Input<std::vector<std::complex<Real> > > _constantQDC;
  Input<std::vector<std::complex<Real> > > _constantQNF;
  Input<std::vector<Real> > _shiftsIn;
  Input<std::vector<Real> > _winsLenIn;
  Input<std::vector<std::vector<Real> > > _freqWinsIn;

 public:
  NSGIConstantQ() {
    declareOutput(_signal, "frame", "the input frame (vector)");
    declareInput(_constantQ, "constantq", "the Non Stationary Gabor transform based constant Q transform of the input frame");
    declareInput(_constantQDC, "constantqdc", "the DC component of the constant Q transform. Needed for the time reconstruction");
    declareInput(_constantQNF, "constantqnf", "the Nyquist Frequency component of the constant Q transform. Needed for the time reconstruction");
    declareInput(_shiftsIn, "windowShifts", "Amount of bins from the center of each frequency window to the base band. Needed for the time reconstruction");
    declareInput(_winsLenIn, "windowLenghts", "Longitudes of the frequency windows used in the transform. Needed for the time reconstruction");
    declareInput(_freqWinsIn, "frequencyWindows", "Frequency windows used in the transform. Needed for the time reconstruction");


    _fft = AlgorithmFactory::create("FFTC");
    _ifft = AlgorithmFactory::create("IFFTC");
  }

  ~NSGIConstantQ() {
    if (_fft) delete _fft;
    if (_ifft) delete _ifft;
  }

  void declareParameters() {
    declareParameter("phaseMode", "'local' to use zero-centered filters. 'global' to use a mapping function [2]", "{local,global}", "global");
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:

  Algorithm* _ifft;
  Algorithm* _fft;

  //Variables for the input parameters
  std::string _phaseMode;


  //windowing vectors
  std::vector<std::vector<Real> > _freqWins;
  std::vector<Real> _shifts;
  std::vector<Real> _winsLen;


  int _binsNum;
  int _NN;
  int _N;

  std::vector<int> _posit;
  std::vector<std::vector<Real> > _dualFreqWins;

  std::vector<std::vector<int> > _win_range;
  std::vector<std::vector<int> > _idx;


  void designDualFrame(const std::vector<Real>& shifts,
                       const std::vector<std::vector<Real> >& freqWins,
                       const std::vector<Real>& winsLen);

};

}
}

#endif // ESSENTIA_NSGICONSTANTQ_H
