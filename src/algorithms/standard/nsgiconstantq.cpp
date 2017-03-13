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

#include "nsgiconstantq.h"
#include "essentia.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* NSGIConstantQ::name = "NSGIConstantQ";
const char* NSGIConstantQ::category = "Standard";
const char* NSGIConstantQ::description = DOC("TODO.\n");


void NSGIConstantQ::configure() {

  _rasterize = parameter("rasterize").toLower();
  _phaseMode = parameter("phaseMode").toLower();
  _normalize = parameter("normalize").toLower();



}


void NSGIConstantQ::compute() {

  const std::vector< std::vector<std::complex<Real> > > & constantQ = _constantQ.get();
  const std::vector<std::complex<Real> >& constantQDC = _constantQDC.get();
  const std::vector<std::complex<Real> >& constantQNF = _constantQNF.get();
  const std::vector<Real>& shifts = _shiftsOut.get();
  const std::vector<Real>& winsLen = _winsLenOut.get();
  std::vector<Real>& signal = _signal.get();


  //add NF and DC components
  std::vector<std::vector<complex<Real> > > CQ;
  CQ = constantQ;
  CQ.push_back(constantQNF);
  CQ.insert(CQ.begin(), constantQDC);

  int N = CQ.size();
  int CH = CQ[1].size();



  _posit.resize(N);
  _posit[0] = shifts[0];

  for (int j=1; j<N; j++) _posit[j] = _posit[j-1] + shifts[j];

  int NN = _posit[N-1];
  std::transform(_posit.begin(), _posit.end(), _posit.begin(),
                  std::bind2nd(std::minus<int>(), shifts[0]));

  std::vector< std::vector<std::complex<Real> > > _fr(CH,std::vector<std::complex<Real> > (NN,0));

  //_fft->configure("size", _inputSize);

  E_INFO(shifts);

}
