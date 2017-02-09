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


  designWindow();
  createCoefficients();

  //todo normalize()

  _fft->configure("size", _inputSize);

  //E_INFO("M:" << _filtersLength);
  //E_INFO("shifts:" << _shifts);
  //E_INFO("g15:" << _freqFilters[14]);
}

void NSGConstantQ::designWindow(){

  //shud this be external parameters?

  Real bwfac = 1;
  int minWin = 4;

  std::vector<Real> cqtbw; //Bandwidths
  std::vector<Real> bw;
  std::vector<Real> posit;

  Real nf = _sr/2;

  // Some exceptions after computing Nyquist frequency.
  if ( _minFrequency < 0 ) {
    throw EssentiaException("NSGConstantQ: 'minimumFrequency' parameter is out of the range (0 - fs/2)");
  }
  if ( _maxFrequency > nf ) {
    throw EssentiaException("NSGConstantQ: 'maximunFrequency' parameter is out of the range (0 - fs/2)");
  }

  Real fftres = _sr / _inputSize;
  Real b = floor( _binsPerOctave * log2(_maxFrequency/_minFrequency));

  _baseFreqs.resize(b + 1);
  cqtbw.resize(b + 1);

  Real Q = pow(2,(1/_binsPerOctave)) - pow(2,(-1/_binsPerOctave));

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


  // Add negative ferquencies
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




  posit.resize( baseFreqsSize );
  for (int j = 0; j <= _binsNum +1; ++j) posit[j] = floor(_baseFreqs[j]);
  for (int j =  _binsNum +2; j < baseFreqsSize; ++j) posit[j] = ceil(_baseFreqs[j]);

  // compute shift in bins
  _shifts.resize( baseFreqsSize );
  _shifts[0] = fmod( - posit[baseFreqsSize-1] , (float) _inputSize );
  for (int j = 1; j< baseFreqsSize; ++j) _shifts[j] = posit[j] - posit[j-1];


  // Fractional mode not implemented (probably not needed)



  std::transform(bw.begin(), bw.end(), bw.begin(),
                  std::bind2nd(std::plus<Real>(), .5));

  _filtersLength.resize(baseFreqsSize);
  copy(bw.begin(),bw.end(), _filtersLength.begin());

  for (int j = 0; j< baseFreqsSize; ++j){
    if (_filtersLength[j] < minWin ) _filtersLength[j] = minWin;
  }

  _freqFilters.resize(baseFreqsSize);



  //Use Windowing to create the requited window filter-bank
  //Todo Maybe we ddon't need this auxiliar vectors?
  std::vector<Real> outputWindow;

  for (int j = 0; j< baseFreqsSize; ++j){

    std::vector<Real> inputWindow(_filtersLength[j], 1);

    _windowing->configure("type", parameter("window").toLower(),
                          "size", _filtersLength[j],
                          "normalized", false);
    _windowing->input("frame").set(inputWindow);
    _windowing->output("frame").set(outputWindow);
    _windowing->compute();


    _freqFilters[j].resize(_filtersLength[j]);
    copy(outputWindow.begin(),outputWindow.end(), _freqFilters[j].begin());

    outputWindow.clear();
    inputWindow.clear();
  }



  // Ceil integer division. Maybe there are matrix operations implemented in essentia?
  std::transform(_filtersLength.begin(), _filtersLength.end(), _filtersLength.begin(),
                  std::bind2nd(std::plus<int>(), - 1));
  std::transform(_filtersLength.begin(), _filtersLength.end(), _filtersLength.begin(),
                  std::bind2nd(std::divides<Real>(), bwfac));
  std::transform(_filtersLength.begin(), _filtersLength.end(), _filtersLength.begin(),
                  std::bind2nd(std::plus<int>(), + 1));

  // Setup Tukey window for 0- and Nyquist-frequency

  // Todo normalize filter

  //E_INFO("filter number: " << _filtersLength);

  for (int j = 0; j <= _binsNum +1; j += _binsNum +1 ){
    if ( _filtersLength[j] > _filtersLength[j+1] ){

      std::vector<Real> inputWindow(_filtersLength[j], 1);
      copy(_freqFilters[j+1].begin(),_freqFilters[j+1].end(), inputWindow.begin() + _filtersLength[j] /2 );
      std::transform(inputWindow.begin(), inputWindow.end(), inputWindow.begin(),
                      std::bind2nd(std::divides<Real>(), 1));
      copy(inputWindow.begin(),inputWindow.end(), _freqFilters[j].begin());
      // E_INFO("new window:");
      // for (int k = 0; k < (int) _filtersLength[j]; ++k) E_INFO(inputWindow[k]);
    }
  }

  _binsNum = baseFreqsSize / 2 - 1;

}


void NSGConstantQ::createCoefficients(){

  if (_rasterize == "full"){

    int rasterizeIdx = _filtersLength.size();

    for (int j = 1; j <= _binsNum; ++j){
      --rasterizeIdx;
      _filtersLength[j] = _filtersLength[_binsNum];
      _filtersLength[rasterizeIdx] = _filtersLength[_binsNum];
    }
    // E_INFO("rasterize: full. New idx:" << _filtersLength);
  }


  if (_rasterize == "piecewise"){

    int octs = ceil(log2(_maxFrequency/ _minFrequency));

    Real temp = ceil(_filtersLength[_binsNum] / pow(2, octs)) * pow(2, octs);
    //E_INFO("temp lengths:" << temp);

    for (int j = 1; j <= (int)_filtersLength.size() ; ++j){
      if (j != _binsNum +1) _filtersLength[j] = temp / ( pow(2, ceil(log2(temp /_filtersLength[j])) -1 ));
    }
    // E_INFO("new lengths:" << _filtersLength);

  }
}

void NSGConstantQ::compute(){

  const std::vector<Real>& signal = _signal.get();
  std::vector< std::vector<Real> >& constantQ = _constantQ.get();

  std::vector<complex<Real> > fft;
  std::vector<int> posit;

  if (signal.size() <= 1) {
    throw EssentiaException("NSGConstantQ: the size of the input signal is not greater than one");
  }

  if (signal.size() !=  _inputSize) {
    E_INFO("NSGConstantQ: The input spectrum size (" << signal.size() << ") does not correspond to the \"inputSize\" parameter (" << _inputSize << "). Recomputing the filter bank.");
    _inputSize = signal.size();
    designWindow();
    createCoefficients();

    //todo normalize()

    _fft->configure("size", _inputSize);

  }

  int N = _shifts.size();

  _fft->input("frame").set(signal);
  _fft->output("fft").set(fft);
  _fft->compute();




  int fill = _shifts[0]  - _inputSize ;

  posit.resize(N);
  posit[0] = 0;

  for (int j = 1; j< N; j++){
    posit[j] = posit[j-1] + _shifts[j] - _shifts[0];
    fill += _shifts[j];
  }
  //E_INFO("posits: " << posit);
  //E_INFO("shifts: " << _shifts);





  //add some zero padding
  //E_INFO("FFT: " <<fft.size());
  std::vector<Real> padding(fill,0.0);
  fft.insert(fft.end(), padding.begin(), padding.end());
  //E_INFO("FFT: " <<fft.size());


  //calculate filter lenghts
  std::vector<int> Lg(_freqFilters.size(),0);

  for (int j = 0;  j < (int)_freqFilters.size(); ++j){
    Lg[j] = _freqFilters[j].size();

    if ( (posit[j] - Lg[j]/2) <= ( _inputSize + fill)/2 ) N = j;
  }
  //E_INFO("Lg: " <<posit);
  //E_INFO("N: " <<N);

  // Prepare indexing vectors and compute the coefficients.
  std::vector<int> idx;
  std::vector<int> win_range;
  std::vector<int> product_idx;
  std::vector<complex <Real> > product;

  constantQ.resize(N);

  for (int j=0; j< N; j++){

    for (int i = ceil( (float) Lg[j]/2.0); i < Lg[j]; i++) idx.push_back(i);
    for (int i = 0; i < ceil( (float) Lg[j]/2); i++) idx.push_back(i);

    float winComp;
    for (int i = -Lg[j]/2 ; i < ceil( (float) Lg[j]/2); i++){
      winComp = (posit[j] + i) % (_inputSize + fill);
      if (winComp < 0) winComp += _inputSize + fill;
      win_range.push_back( winComp);
    }


    if (_filtersLength[j] < Lg[j]){
      int col = ceil(Lg[j]/_filtersLength[j]);

      std::vector<Real> idx(col*_filtersLength[j],0.0);

      //todo implement non-painles case
    }
    else{
      for (int i = _filtersLength[j] - (Lg[j] )/2 ; i < _filtersLength[j] + Lg[j]/2; i++) product_idx.push_back( fmod(i, _filtersLength[j]));
      product.resize(_filtersLength[j]);
      std::fill(product.begin(), product.end(), 0);
      if  (j == 0){
        E_INFO("Lg[j]: " <<Lg[j]);
        E_INFO("_filtersLength[j]: " <<_filtersLength[j]);
        E_INFO("idx: " <<idx);
        E_INFO("win_range: " <<win_range);
        E_INFO("product:idx: " <<product_idx);
        for (int i = _filtersLength[j] - (Lg[j])/2 ; i < _filtersLength[j] + Lg[j]/2; i++) E_INFO("idx: " <<i);
      }
      /*
      E_INFO("_freqFilters: " );
      for (int i = 0 ; i < idx.size() ; i++) E_INFO( _freqFilters[j][idx[i]]);
      E_INFO("fft: " );
      for (int i = 0 ; i < idx.size() ; i++) E_INFO( fft[win_range[i]]);
      //E_INFO("fft: " << fft[win_range[i]]);
      */
      for (int i = 0 ; i < idx.size() ; i++){
          product[idx[i]] = fft[win_range[i]] * _freqFilters[j][idx[i]];

          //todo ifft
          _ifft->configure("size",_filtersLength[j]);
          _ifft->input("fft").set(product);
          _ifft->output("frame").set(constantQ[j]);
          _ifft->compute();

      }

    }

    idx.clear();
    win_range.clear();
    product_idx.clear();
    product.clear();

  }


  //E_INFO("win_range: " <<win_range);
}




