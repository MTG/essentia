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
const char* NSGIConstantQ::description = DOC("This algorithm computes an inverse constant Q transform using non stationary Gabor frames and returns a complex time-frequency representation of the input vector.\n"
"The implementation is inspired by the toolbox described in [1]."
"\n"
"References:\n"
"  [1] Schörkhuber, C., Klapuri, A., Holighaus, N., \& Dörfler, M. (n.d.). A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.");


void NSGIConstantQ::configure() {
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

  if (_inputSize % 2) {
    _oddInput = true;
    _inputSize++;
  } else {
    _oddInput = false;
  }

  designWindow();
  createCoefficients();
  normalize();
  designDualFrame();

  int shiftsSize = _shifts.size();
  _N = shiftsSize/2 + 1;

  _posit.resize(shiftsSize);
  _posit[0] = _shifts[0];

  for (int j=1; j<shiftsSize; j++) _posit[j] = _posit[j-1] + _shifts[j];

  _NN = _posit[shiftsSize-1];

  std::transform(_posit.begin(), _posit.end(), _posit.begin(),
                   std::bind2nd(std::minus<int>(), _shifts[0]));
}

void NSGIConstantQ::designWindow() {
  std::vector<Real> cqtbw; //Bandwidths
  std::vector<Real> bw;
  std::vector<Real> posit;

  Real nf = _sr / 2;


  // Some exceptions after computing Nyquist frequency.
  if (_minFrequency < 0) {
    throw EssentiaException("NSGIConstantQ: 'minimumFrequency' parameter is out of the range (0 - fs/2)");
  }
  if (_maxFrequency > nf) {
    throw EssentiaException("NSGIConstantQ: 'maximunFrequency' parameter is out of the range (0 - fs/2)");
  }
  if (_minFrequency >= _maxFrequency) {
    throw EssentiaException("NSGIConstantQ: 'minimumFrequency' has to be lower than 'maximunFrequency'");
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
      _freqWins[j] = std::vector<Real>(_winsLen[j], 1);

      copy(_freqWins[j+1].begin(),_freqWins[j+1].end(), _freqWins[j].begin() + _winsLen[j]/2 - _winsLen[j+1]/2 );


      //copy(inputWindow.begin(),inputWindow.end(), _freqWins[j].begin());

      std::transform( _freqWins[j].begin(),  _freqWins[j].end(),  _freqWins[j].begin(),
                      std::bind2nd(std::divides<Real>(), sqrt(_winsLen[j] )));

    }
  }

  _binsNum = baseFreqsSize / 2 - 1;
}


void NSGIConstantQ::createCoefficients() {

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

  // filters have to be even as Essentia odd size FFT is not implemented.
  for (int j=0; j<(int)_winsLen.size(); j++) _winsLen[j] += (_winsLen[j] % 2);
}


void NSGIConstantQ::normalize() {
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

void NSGIConstantQ::compute() {

  const std::vector< std::vector<std::complex<Real> > > & constantQ = _constantQ.get();
  const std::vector<std::complex<Real> >& constantQDC = _constantQDC.get();
  const std::vector<std::complex<Real> >& constantQNF = _constantQNF.get();
  const std::vector<int>& winsLen = _winsLen;
  const std::vector< std::vector<Real> >& freqWins = _freqWins;
  std::vector<Real>& signal = _signal.get();

  //add NF and DC components
  std::vector<std::vector<complex<Real> > > CQ;
  CQ = constantQ;
  CQ.push_back(constantQNF);
  CQ.insert(CQ.begin(), constantQDC);

  if (CQ.size() != _N) {
    throw EssentiaException("NSGIConstantQ: input data donesn't match the shape of the generated dual frames. Make sure to configure this algorithm with the same parameters used in the analysis by NSGConstantQ");
  }



  // Dual frames only have to be computed the first time.
/*
  if (_shifts != shifts || _winsLen != winsLen || _freqWins != freqWins ){

    designDualFrame(shifts, freqWins, winsLen);


    _N = CQ.size();
    int shiftsSize = shifts.size();

    _posit.resize(shiftsSize);
    _posit[0] = shifts[0];

    for (int j=1; j<shiftsSize; j++) _posit[j] = _posit[j-1] + shifts[j];

    _NN = _posit[shiftsSize-1];

    std::transform(_posit.begin(), _posit.end(), _posit.begin(),
                     std::bind2nd(std::minus<int>(), shifts[0]));

    _shifts = shifts;
    _winsLen = winsLen;
    _freqWins = freqWins;
  }
*/

  std::vector<std::complex<Real> >  fr(_NN, (complex<Real>)0);
  std::vector<int> temp_idx;
  std::vector<std::complex<Real> > temp;

  for (int j=0; j<_N; j++){
    int Lg = freqWins[j].size();

    for (int i=(int)winsLen[j]-(Lg )/2; i<(int)winsLen[j]+int(Real(Lg)/2 + .5); i++) temp_idx.push_back( fmod(i, (int)winsLen[j]));


    _fft->configure("size", (int)winsLen[j],
                    "negativeFrequencies", true);
    _fft->input("frame").set(CQ[j]);
    _fft->output("fft").set(temp);
    _fft->compute();

    std::transform(temp.begin(), temp.end(), temp.begin(),
                    std::bind2nd(std::multiplies<complex<Real> >(), winsLen[j]));

    // Phase shift
    if (_phaseMode == "global") {
      int displace = (_posit[j] - (_posit[j] / (int)winsLen[j] * (int)winsLen[j])) % temp.size();

      std::rotate(temp.begin(),
                  temp.begin() + displace, // this will be the new first element
                  temp.end());
    }

    for (int i=0; i<(int)_win_range[j].size(); i++){
      fr[_win_range[j][i]] += temp[temp_idx[i]] * _dualFreqWins[j][_idx[j][i]] ;
    }

    temp.clear();
    temp_idx.clear();
  }

  int NyquistBin = _NN/2;
  int count = 1;
  for (int i= NyquistBin -1; i>0; i--){
    fr[NyquistBin + count] = std::conj(fr[i]);
    count++;
  }

  std::vector<std::complex<Real> > output;
  _ifft->configure("size", _NN);
  _ifft->input("fft").set(fr);
  _ifft->output("frame").set(output);
  _ifft->compute();


  std::reverse(output.begin()+1, output.end());

  std::transform(output.begin(), output.end(), output.begin(),
                  std::bind2nd(std::divides<complex<Real> >(), _NN));

  signal.resize(_NN);
  for (int i=0; i<_NN; i++){
    signal[i] = std::real(output[i]);
  }

  // If the algorithm was configured with an odd size
  // it means that the last sample was added on the
  // forward pass and should be removed in order to
  // get the original size.
  if (_oddInput) {
    signal.pop_back();
  }
}


void NSGIConstantQ::designDualFrame(){

  _posit.clear();
  _win_range.clear();
  _idx.clear();


  Real eps = std::numeric_limits<Real>::epsilon();
  int N = _shifts.size();


  _posit.resize(N);
  _posit[0] = _shifts[0];

  for (int j=1; j<N; j++) _posit[j] = _posit[j-1] + _shifts[j];

  int Ls = _posit[N-1];

  std::transform(_posit.begin(), _posit.end(), _posit.begin(),
                  std::bind2nd(std::minus<int>(), _shifts[0]));

  std::vector<Real> diagonal(Ls, 0.0);


  _win_range.resize(N);
  _idx.resize(N);
  for (int j = 0; j<N; j++){
    int Lg = _freqWins[j].size();

    for (int i = ceil( (float) Lg/2.0); i < Lg; i++) _idx[j].push_back(i);
    for (int i = 0; i < ceil( (float) Lg/2); i++) _idx[j].push_back(i);

    float winComp;
    for (int i = -Lg/2; i < ceil((float) Lg / 2); i++){
      winComp = (_posit[j] + i) % Ls;
      if (winComp < 0){
        winComp = Ls + winComp;
      }
      _win_range[j].push_back( abs(winComp));
    }

    for (int i=0; i<(int)_win_range[j].size(); i++){
      diagonal[_win_range[j][i]] += pow(_freqWins[j][_idx[j][i]], 2) * _winsLen[j] + eps;
    }
  }

  _dualFreqWins = _freqWins;

  for (int j = 0; j<N; j++){

    for (int i=0; i<(int)_win_range[j].size(); i++){
      _dualFreqWins[j][_idx[j][i]] = _dualFreqWins[j][_idx[j][i]] / diagonal[_win_range[j][i]];
    }
  }
}

