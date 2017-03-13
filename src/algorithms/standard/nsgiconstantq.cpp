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
  const std::vector<std::vector<Real> >& freqWins = _freqWins.get();
  std::vector<Real>& signal = _signal.get();

  //add NF and DC components
  std::vector<std::vector<complex<Real> > > CQ;
  CQ = constantQ;
  CQ.push_back(constantQNF);
  CQ.insert(CQ.begin(), constantQDC);

  int N = CQ.size();
  int shiftsSize = shifts.size();

  _posit.resize(shiftsSize);
  _posit[0] = shifts[0];

  for (int j=1; j<shiftsSize; j++) _posit[j] = _posit[j-1] + shifts[j];

  int NN = _posit[shiftsSize-1];

  std::transform(_posit.begin(), _posit.end(), _posit.begin(),
                  std::bind2nd(std::minus<int>(), shifts[0]));

  std::vector<std::complex<Real> >  fr(NN, (complex<Real>)0);

  std::vector<int> win_range;
  std::vector<int> idx;
  std::vector<std::complex<Real> > temp;

  for (int j=0; j<N; j++){
    int Lg = freqWins[j].size();

    for (int i = ceil( (float) Lg/2.0); i < Lg; i++) idx.push_back(i);
    for (int i = 0; i < ceil( (float) Lg/2); i++) idx.push_back(i);

    float winComp;
    for (int i = -Lg/2; i < ceil((float) Lg / 2); i++){
      winComp = (_posit[j] + i) % NN;
      if (winComp < 0){
        winComp = (NN) + winComp;
      }

      win_range.push_back( abs(winComp));
    }

    _fft->configure("size", (int)winsLen[j]);
    _fft->input("frame").set(CQ[j]);
    _fft->output("fft").set(temp);
    _fft->compute();

    for (int i=temp.size()-2; i>0; i--) temp.push_back(temp[i]);

    std::transform(temp.begin(), temp.end(), temp.begin(),
                    std::bind2nd(std::divides<complex<Real> >(), winsLen[j]));

    // E_INFO(win_range);
    // E_INFO(idx);
    // E_INFO(temp);
    // E_INFO(freqWins[j]);
    for (int i=0; i<(int)win_range.size(); i++){
      fr[win_range[i]] += temp[idx[i]] * freqWins[j][idx[i]];
    }


    // E_INFO(temp);
    idx.clear();
    win_range.clear();
  }
  int NyquistBin = NN/2 + 1;

  for (int i=1; i<(int)NyquistBin; i++){
    fr[NN-i-1] = std::conj(fr[i]);
  }

  std::vector<std::complex<Real> > output;
  _ifft->configure("size", NN);
  _ifft->input("fft").set(fr);
  _ifft->output("frame").set(output);
  _ifft->compute();

  signal.resize(NN);
  for (int i=0; i<NN; i++){
    signal[i] = std::real(output[i]);
  }

}
