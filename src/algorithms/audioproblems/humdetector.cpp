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

#include "humdetector.h"
#include <algorithm> // sort
#include "essentiamath.h"

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* HumDetector::name = essentia::standard::HumDetector::name;
const char* HumDetector::category = essentia::standard::HumDetector::category;
const char* HumDetector::description = essentia::standard::HumDetector::description;


template< typename T >
typename std::vector<T>::iterator 
   HumDetector::insertSorted( std::vector<T> & vec, T const& item )
{
    return vec.insert
        ( 
            std::upper_bound( vec.begin(), vec.end(), item ),
            item 
        );
}

template <typename T>
vector<size_t> HumDetector::sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

HumDetector::HumDetector() : AlgorithmComposite() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _decimator                = factory.create("Resample");
  _lowPass                  = factory.create("LowPass");
  _frameCutter              = factory.create("FrameCutter");
  _welch                    = factory.create("PowerSpectrum");
  _Smoothing                = standard::AlgorithmFactory::create("MedianFilter");
  _spectralPeaks            = standard::AlgorithmFactory::create("SpectralPeaks");
  // _timeSmoothing            = factory.create("MedianFilter");

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_rMatrix, "r", "r");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_amplitudes, "amplitudes", "humming tones amplitudes");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  // Connect input proxy
  _signal >> _decimator->input("signal");

  _decimator->output("signal").setBufferType(BufferUsage::forLargeAudioStream);
  
  _decimator->output("signal") >> _lowPass->input("signal");

  _lowPass->output("signal").setBufferType(BufferUsage::forLargeAudioStream);

  _lowPass->output("signal") >> _frameCutter->input("signal");

  _frameCutter->output("frame") >> _welch->input("signal");


  _welch->output("powerSpectrum") >> PC(_pool, "psd");

  _network = new scheduler::Network(_decimator);
}

HumDetector::~HumDetector() {
  delete _network;
}


void HumDetector::configure() {

  _outSampleRate = 2000.f;
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = int(round(parameter("hopSize").toReal() * _outSampleRate));
  _frameSize = nextPowerTwo(int(round(parameter("frameSize").toReal() * _outSampleRate)));
  _timeWindow = int(round(parameter("timeWindow").toReal() * _outSampleRate / _hopSize));
  _Q0 = parameter("Q0").toReal();
  _Q1 = parameter("Q1").toReal();

  _Q0sample = (uint)(_Q0 * _timeWindow + 0.5);
  _Q1sample = (uint)(_Q1 * _timeWindow + 0.5);

  _medianFilterSize = _frameSize * 60 / _outSampleRate;
  _medianFilterSize += (_medianFilterSize + 1) % 2;

  _decimator->configure("inputSampleRate", _sampleRate,
                        "outputSampleRate", _outSampleRate);
  
  _lowPass->configure("sampleRate",_outSampleRate,
                      "cutoffFrequency", 900.f);

  _frameCutter->configure("frameSize",_frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "keep");

  _welch->configure("size",_frameSize);

  _spectralPeaks->configure("sampleRate", _outSampleRate,
                            "magnitudeThreshold", 10.f);
  // _timeSmoothing->configure();

}


AlgorithmStatus HumDetector::process() {
  if (!shouldStop()) return PASS;

  if (!_pool.contains<vector<vector<Real> > >("psd")) {
    // do not push anything in the case of empty signal
    E_WARNING("HumDetector: empty input signal");
    return FINISHED;
  }

  const vector<vector<Real> >& psd = _pool.value<vector<vector<Real> > >("psd");

  _spectSize = psd[0].size();
  _timeStamps = psd.size();
  _iterations = _timeStamps - _timeWindow + 1;
  std::vector<vector<Real> >psdWindow(_spectSize, vector<Real>(_timeWindow, 0.f));
  std::vector<vector<size_t> >psdIdxs(_spectSize, vector<size_t>(_timeWindow, 0));


  std::vector<vector<Real> >r(_spectSize, vector<Real>(_iterations, 0.0));

  Real Q0, Q1;
  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = 0; j < _timeWindow; j++)
      psdWindow[i][j] = psd[j][i];

    psdIdxs[i] = sort_indexes(psdWindow[i]);
    // sort(psdWindow[i].begin(), psdWindow[i].end());
    Q0 = psdWindow[i][psdIdxs[i][_Q0sample]];
    Q1 = psdWindow[i][psdIdxs[i][_Q1sample]];
    
    r[i][0] = Q0 / Q1;
  }

  // std::vector<Real>::iterator idxIt;
  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = _timeWindow; j < _timeStamps; j++) {
      rotate(psdWindow[i].begin(), psdWindow[i].begin() + 1, psdWindow[i].end());
      // idxIt = insertSorted(psdWindow[i], psd[j][i]);
      // uint pos = std::distance(psdWindow[0].begin(),idxIt);
      psdWindow[i][_timeWindow - 1] = psd[j][i];
      psdIdxs[i] = sort_indexes(psdWindow[i]);

      Q0 = psdWindow[i][psdIdxs[i][_Q0sample]];
      Q1 = psdWindow[i][psdIdxs[i][_Q1sample]];

      r[i][j - _timeWindow + 1] = Q0 / Q1;
      }
  }
  vector<Real> rSpec = vector<Real>(_spectSize, 0.f);
  vector<Real> filtered = vector<Real>(_spectSize, 0.f);
  for (uint j = 0; j < _iterations; j++) {
    for (uint i = 0; i < _spectSize; i++)
      rSpec[i] = r[i][j];

    _Smoothing->configure("kernelSize", _medianFilterSize);
    _Smoothing->input("array").set(rSpec);
    _Smoothing->output("filteredArray").set(filtered);
    _Smoothing->compute();
    
    for (uint i = 0; i < _spectSize; i++)
      r[i][j] -= filtered[i];
  }

  uint kernerSize = (uint)_timeWindow + (((uint)_timeWindow + 1) % 2);
  for (uint i = 0; i < _spectSize; i++) {
    _Smoothing->configure("kernelSize", kernerSize);
    _Smoothing->input("array").set(r[i]);
    _Smoothing->output("filteredArray").set(filtered);
    _Smoothing->compute();

    for (uint j = 0; j < _iterations; j++)
        r[i][j] = filtered[j];
  }

  _rMatrix.push(vecvecToArray2D(r));

  vector<Real> aux(1, 1.0);

  _amplitudes.push(aux);
  _frequencies.push(aux);
  _starts.push(aux);
  _ends.push(aux);

  
  vector<Real> frequencies, magnitudes;
  for (uint j = 0; j < _iterations; j++) {
    for (uint i = 0; i < _spectSize; i++) 
      rSpec[i] = r[i][j];

    _spectralPeaks->input("spectrum").set(rSpec);
    _spectralPeaks->output("frequencies").set(frequencies);
    _spectralPeaks->output("magnitudes").set(magnitudes);
    _spectralPeaks->compute();

    

    // for (uint k = 0; k < frequencies.size(); k++) {
    // _frequencies.push(frequencies);
    // _amplitudes.push(magnitudes);
    // }
  }

  return FINISHED;
}

void HumDetector::reset() {
  AlgorithmComposite::reset();
  _pool.remove("psd");
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* HumDetector::name = "HumDetector";
const char* HumDetector::category = "Audio Problems";
const char* HumDetector::description = DOC("");


HumDetector::HumDetector() {
  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_rMatrix, "r", "r");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_amplitudes, "amplitudes", "humming tones amplitudes");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  createInnerNetwork();
}

HumDetector::~HumDetector() {
  delete _network;
}

void HumDetector::configure() {
  _humDetector->configure(INHERIT("sampleRate"), INHERIT("hopSize"),
                          INHERIT("frameSize"), INHERIT("timeWindow"));
}


void HumDetector::createInnerNetwork() {
  _humDetector = streaming::AlgorithmFactory::create("HumDetector");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _humDetector->input("signal");
  _humDetector->output("r")    >> PC(_pool, "r");
  _humDetector->output("frequencies")    >> PC(_pool, "frequencies");
  _humDetector->output("amplitudes")    >> PC(_pool, "amplitudes");
  _humDetector->output("starts")   >> PC(_pool, "starts");
  _humDetector->output("ends")        >> PC(_pool, "ends");

  _network = new scheduler::Network(_vectorInput);
}

void HumDetector::compute() {
  const vector<Real>& signal = _signal.get();
  if (!signal.size()) {
    throw EssentiaException("HumDetector: empty input signal");
  }

  _vectorInput->setVector(&signal);
  _network->run();

  TNT::Array2D<Real>& rMatrix = _rMatrix.get();
  vector<Real>& frequencies = _frequencies.get();
  vector<Real>& amplitudes = _amplitudes.get();
  vector<Real>& starts = _starts.get();
  vector<Real>& ends = _ends.get();


  rMatrix = _pool.value<vector<TNT::Array2D<Real> > >("r")[0];


  frequencies = _pool.value<vector<Real> >("frequencies");
  amplitudes = _pool.value<vector<Real> >("amplitudes");
  starts = _pool.value<vector<Real> >("starts");
  ends = _pool.value<vector<Real> >("ends");

  // reset();
}

void HumDetector::reset() {
  _network->reset();
  _pool.remove("r");
  _pool.remove("frequencies");
  _pool.remove("amplitudes");
  _pool.remove("starts");
  _pool.remove("ends");
  //_pool.remove("ends");
  //_pool.remove("shortTermLoudnessMax");
}

} // namespace standard
} // namespace essentia
