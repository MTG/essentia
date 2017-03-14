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

#include "nsgconstantq.h"
#include "essentia.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* NSGConstantQ::name = "NSGConstantQ";
const char* NSGConstantQ::category = "Standard";
const char* NSGConstantQ::description = DOC("TODO.\n");


void NSGConstantQ::configure() {
  _sr = parameter("sampleRate").toReal();
  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();


  _binsPerOctave = parameter("binsPerOctave").toReal();
  _gamma = parameter("gamma").toReal();
  _inputSize = parameter("inputSize").toInt();

  _rasterize = parameter("rasterize").toLower();
  _phaseMode = parameter("phaseMode").toLower();
  _normalize = parameter("normalize").toLower();
  _minimumWindow = parameter("minimumWindow").toInt();
  _windowSizeFactor = parameter("windowSizeFactor").toInt();

  _INSQConstantQdata = parameter("INSQConstantQdata").toBool();

  designWindow();
  createCoefficients();
  normalize();

  _fft->configure("size", _inputSize);

}

void NSGConstantQ::designWindow() {
  std::vector<Real> cqtbw; //Bandwidths
  std::vector<Real> bw;
  std::vector<Real> posit;

  Real nf = _sr / 2;


  // Some exceptions after computing Nyquist frequency.
  if (_minFrequency < 0) {
    throw EssentiaException("NSGConstantQ: 'minimumFrequency' parameter is out of the range (0 - fs/2)");
  }
  if (_maxFrequency > nf) {
    throw EssentiaException("NSGConstantQ: 'maximunFrequency' parameter is out of the range (0 - fs/2)");
  }
  if (_minFrequency >= _maxFrequency) {
    throw EssentiaException("NSGConstantQ: 'minimumFrequency' has to be lower than 'maximunFrequency'");
  }


  Real fftres = _sr / _inputSize;
  Real Q = pow(2,(1/_binsPerOctave)) - pow(2,(-1/_binsPerOctave));
  Real b = floor(_binsPerOctave * log2(_maxFrequency/_minFrequency));

  _baseFreqs.resize(b + 1);
  cqtbw.resize(b + 1);


  // compute bandwidth for each bin
  for (int j=0; j<= b; ++j) {
    _baseFreqs[j] = ( _minFrequency * pow(2,j / _binsPerOctave) ) ;
    cqtbw[j] = Q * _baseFreqs[j] + _gamma;
  }


  // check if the first and last bins are within the bounds.
  if ( _baseFreqs[0] - cqtbw[0] / 2 < 0 ) {
    E_INFO("NSGConstantQ: Attempted to create a band with a low bound of " << _baseFreqs[0] - cqtbw[0] << "Hz");
    throw EssentiaException("NSGConstantQ: Attempted to create a filter below frequency 0");
  }
  if ( _baseFreqs[b] + cqtbw[b] / 2 > nf ) {
    _baseFreqs.pop_back();
    E_INFO("NSGConstantQ: Removing last bin because it was over the Nyquist Frequency");
  }


  _binsNum = _baseFreqs.size();
  _baseFreqs.insert(_baseFreqs.begin(),0.0);
  _baseFreqs.push_back(nf);


  // Add negative frequencies
  for (int j = _binsNum ; j > 0; --j) _baseFreqs.push_back(_sr -_baseFreqs[j]);

  int baseFreqsSize = (int) _baseFreqs.size();

  bw.push_back(2 * _minFrequency );
  bw.insert(bw.end(),cqtbw.begin(), cqtbw.end());
  bw.push_back(_baseFreqs[_binsNum+2] - _baseFreqs[_binsNum]);
  for (int j = cqtbw.size() -1; j >= 0; --j) bw.push_back(cqtbw[j]);


  //bins to Hz
  std::transform(_baseFreqs.begin(), _baseFreqs.end(), _baseFreqs.begin(),
                  std::bind2nd(std::divides<Real>(), fftres));

  std::transform(bw.begin(), bw.end(), bw.begin(),
                  std::bind2nd(std::divides<Real>(), fftres));


  posit.resize(baseFreqsSize);
  for (int j = 0; j <= _binsNum +1; ++j) posit[j] = floor(_baseFreqs[j]);
  for (int j =  _binsNum +2; j < baseFreqsSize; ++j) posit[j] = ceil(_baseFreqs[j]);

  // compute shift in bins
  _shifts.resize( baseFreqsSize );
  _shifts[0] = fmod( - posit[baseFreqsSize-1] , (float) _inputSize );
  for (int j = 1; j< baseFreqsSize; ++j) _shifts[j] = posit[j] - posit[j-1];


  // Fractional mode not implemented (probably not needed)



  std::transform(bw.begin(), bw.end(), bw.begin(),
                  std::bind2nd(std::plus<Real>(), .5));

  _winsLen.resize(baseFreqsSize);
  copy(bw.begin(),bw.end(), _winsLen.begin());

  for (int j = 0; j< baseFreqsSize; ++j){
    if (_winsLen[j] < _minimumWindow ) _winsLen[j] = _minimumWindow;
  }

  _freqWins.resize(baseFreqsSize);


  //Use Windowing to create the requited window filter-bank
  for (int j = 0; j< baseFreqsSize; ++j){

    std::vector<Real> inputWindow(_winsLen[j], 1);

    _windowing->configure("type", parameter("window").toLower(),
                          "size", _winsLen[j],
                          "normalized", false,
                          "zeroPhase", false);
    _windowing->input("frame").set(inputWindow);
    _windowing->output("frame").set( _freqWins[j]);
    _windowing->compute();


    inputWindow.clear();
  }


  // Ceil integer division. Maybe there are matrix operations implemented in essentia?
  std::transform(_winsLen.begin(), _winsLen.end(), _winsLen.begin(),
                  std::bind2nd(std::plus<int>(), - 1));
  std::transform(_winsLen.begin(), _winsLen.end(), _winsLen.begin(),
                  std::bind2nd(std::divides<Real>(), _windowSizeFactor));
  std::transform(_winsLen.begin(), _winsLen.end(), _winsLen.begin(),
                  std::bind2nd(std::plus<int>(), + 1));


  // Setup Tukey window for 0- and Nyquist-frequency
  for (int j = 0; j <= _binsNum +1; j += _binsNum +1) {
    if ( _winsLen[j] > _winsLen[j+1] ) {
      std::vector<Real> inputWindow(_winsLen[j], 1);

      copy(_freqWins[j+1].begin(),_freqWins[j+1].end(), inputWindow.begin() + _winsLen[j] /2 );

      std::transform(inputWindow.begin(), inputWindow.end(), inputWindow.begin(),
                      std::bind2nd(std::divides<Real>(), 1));

      copy(inputWindow.begin(),inputWindow.end(), _freqWins[j].begin());

      std::transform( _freqWins[j].begin(),  _freqWins[j].end(),  _freqWins[j].begin(),
                      std::bind2nd(std::divides<Real>(), sqrt(_winsLen[j] )));

    }
  }

  _binsNum = baseFreqsSize / 2 - 1;

  _shiftsReal.resize(_shifts.size());
  for (int j=0; j<_shifts.size(); j++) _shiftsReal[j] = Real(_shifts[j]);
}


void NSGConstantQ::createCoefficients() {

  if (_rasterize == "full"){
    int rasterizeIdx = _winsLen.size();

    for (int j = 1; j <= _binsNum; ++j){
      --rasterizeIdx;
      _winsLen[j] = _winsLen[_binsNum];
      _winsLen[rasterizeIdx] = _winsLen[_binsNum];
    }
  }

  if (_rasterize == "piecewise"){
    int octs = ceil(log2(_maxFrequency/ _minFrequency));
    Real temp = ceil(_winsLen[_binsNum] / pow(2, octs)) * pow(2, octs);

    for (int j = 1; j < (int)_winsLen.size() ; ++j){
      if (j != _binsNum +1) _winsLen[j] = temp / (pow(2, ceil(log2(temp / _winsLen[j])) -1 ));
    }
  }

  _winsLenReal.resize(_winsLen.size());
  for (int j=0; j<_winsLen.size(); j++) _winsLenReal[j] = Real(_winsLen[j]);
}


void NSGConstantQ::normalize() {
  std::vector<Real> normalizeWeights(_binsNum+2, 1);

  if (_normalize == "sine"){
    copy(_winsLen.begin(), _winsLen.begin() + _binsNum+2, normalizeWeights.begin());

    std::transform(normalizeWeights.begin(), normalizeWeights.end(), normalizeWeights.begin(),
                    std::bind2nd(std::multiplies<Real>(), 2 / Real(_inputSize)));

    for (int j = _binsNum; j > 0; --j) normalizeWeights.push_back(normalizeWeights[j]);
  }


  if (_normalize == "impulse") {
    copy(_winsLen.begin(), _winsLen.begin() + _binsNum+2, normalizeWeights.begin());

    for(int j = 0; j < _binsNum +2; j++){
      normalizeWeights[j] = normalizeWeights[j] * 2 / Real(_freqWins[j].size());
    }

    for (int j = _binsNum; j > 0; --j) normalizeWeights.push_back(normalizeWeights[j]);
  }


  for (int j = 0; j < (int)_freqWins.size(); j++){
    std::transform(_freqWins[j].begin(), _freqWins[j].end(), _freqWins[j].begin(),
                    std::bind2nd(std::multiplies<Real>(), normalizeWeights[j]));
  }
}


void NSGConstantQ::compute() {
  const std::vector<Real>& signal = _signal.get();
  std::vector<std::vector<complex<Real> > >& constantQ = _constantQ.get();
  std::vector<complex<Real> >& constantQDC = _constantQDC.get();
  std::vector<complex<Real> >& constantQNF = _constantQNF.get();
  std::vector<Real>& shiftsOut = _shiftsOut.get();
  std::vector<Real>& winsLenOut = _winsLenOut.get();
  std::vector< std::vector<Real> >&  freqWinsOut = _freqWinsOut.get();

  std::vector<complex<Real> > fft;
  std::vector<int> posit;

  if (signal.size() <= 1) {
    throw EssentiaException("NSGConstantQ: the size of the input signal is not greater than one");
  }

  // Check input. If different shape reconfigure the algorithm
  if (signal.size() !=  _inputSize) {
    E_INFO("NSGConstantQ: The input spectrum size (" << signal.size() << ") does not correspond to the \"inputSize\" parameter (" << _inputSize << "). Recomputing the filter bank.");
    _inputSize = signal.size();

    designWindow();
    createCoefficients();
    normalize();

    _fft->configure("size", _inputSize);
  }

  int N = _shifts.size();

  _fft->input("frame").set(signal);
  _fft->output("fft").set(fft);
  _fft->compute();

  int fill = _shifts[0]  - _inputSize ;

  posit.resize(N);
  posit[0] = _shifts[0];

  for (int j=1; j<N; j++) {
    posit[j] = posit[j-1] + _shifts[j];
    fill += _shifts[j];
  }

  std::transform(posit.begin(), posit.end(), posit.begin(),
                  std::bind2nd(std::minus<int>(), _shifts[0]));

  //add some zero padding if needed
  std::vector<Real> padding(fill,0.0);
  fft.insert(fft.end(), padding.begin(), padding.end());

  // extract filter lengths
  std::vector<int> Lg(_freqWins.size(),0);

  for (int j = 0;  j < (int)_freqWins.size(); ++j) {
    Lg[j] = _freqWins[j].size();

    if ((posit[j] - Lg[j]/2) <= float(_inputSize + fill)/2) N = j+1;
  }

  // Prepare indexing vectors and compute the coefficients.
  std::vector<int> idx;
  std::vector<int> win_range;
  std::vector<int> product_idx;
  std::vector<complex <Real> > product;
  constantQ.resize(N);

  // The actual Gabor transform
  for (int j=0; j<N; j++){

    for (int i = ceil( (float) Lg[j]/2.0); i < Lg[j]; i++) idx.push_back(i);
    for (int i = 0; i < ceil( (float) Lg[j]/2); i++) idx.push_back(i);

    float winComp;
    for (int i = -Lg[j]/2; i < ceil((float) Lg[j] / 2); i++){
      winComp = (posit[j] + i) % (_inputSize + fill);
      if (winComp >= fft.size()){
        winComp = (_inputSize + fill) - winComp;
      }

      win_range.push_back( abs(winComp));
    }


    if (_winsLen[j] < Lg[j]) {
      throw EssentiaException("NSGConstantQ: non painless frame found. This case is currently not supported.");
      // TODO implement non-painless case
    }
    else {
      for (int i = _winsLen[j] - (Lg[j] )/2 ; i < _winsLen[j] + int( Real(Lg[j])/2 + .5); i++) product_idx.push_back( fmod(i, _winsLen[j]));

      product.resize(_winsLen[j]);
      std::fill(product.begin(), product.end(), 0);

      for (int i = 0; i < idx.size(); i++) {
        product[product_idx[i]] = fft[win_range[i]] * _freqWins[j][idx[i]];
      }

      // Circular shift in order to get the global phase representation
      if (_phaseMode == "global") {
        int displace = (posit[j] - ((posit[j] / _winsLen[j]) * _winsLen[j])) % product.size();

        std::rotate(product.begin(),
                    product.end() - displace, // this will be the new first element
                    product.end());
      }

      _ifft->configure("size",_winsLen[j]);
      _ifft->input("fft").set(product);
      _ifft->output("frame").set(constantQ[j]);
      _ifft->compute();

      std::transform(constantQ[j].begin(), constantQ[j].end(), constantQ[j].begin(),
                      std::bind2nd(std::divides<complex<Real> >(), constantQ[j].size()));
    }

    idx.clear();
    win_range.clear();
    product_idx.clear();
    product.clear();
  }


  if (_INSQConstantQdata){
    constantQDC.resize(constantQ[0].size());
    copy(constantQ[0].begin(),constantQ[0].end(),constantQDC.begin());

    constantQNF.resize(constantQ[N-1].size());
    copy(constantQ[N-1].begin(),constantQ[N-1].end(),constantQNF.begin());

    shiftsOut = _shiftsReal;
    winsLenOut = _winsLenReal;
    freqWinsOut = _freqWins;
  }

  // boundary bins are removed from the main output
  constantQ.pop_back();
  constantQ.erase(constantQ.begin());
}




