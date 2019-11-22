/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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
#include "poolstorage.h"

using namespace std;


namespace essentia {
namespace streaming {

const char* HumDetector::name = essentia::standard::HumDetector::name;
const char* HumDetector::category = essentia::standard::HumDetector::category;
const char* HumDetector::description = essentia::standard::HumDetector::description;


template< typename T >
typename std::vector<T>::iterator 
   HumDetector::insertSorted(std::vector<T> & vec, T const& item) {
    return vec.insert(std::upper_bound(vec.begin(), 
                                       vec.end(), 
                                       item), item);
}


template <typename T>
vector<size_t> HumDetector::sort_indexes(const vector<T> &v) {

  // Initialize original index locations.
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // Sort indexes based on comparing values in v.
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


Real HumDetector::centBinToFrequency(Real cent, Real reff, Real binsInOctave) {
  return pow(2.f, (cent - reff) / binsInOctave);
}


HumDetector::HumDetector() : AlgorithmComposite() {
  AlgorithmFactory& factory   = AlgorithmFactory::instance();
  _decimator                  = factory.create("Resample");
  _lowPass                    = factory.create("LowPass");
  _frameCutter                = factory.create("FrameCutter");
  _welch                      = factory.create("Welch");
  _Smoothing                  = standard::AlgorithmFactory::create("MedianFilter");
  _spectralPeaks              = standard::AlgorithmFactory::create("SpectralPeaks");
  _pitchSalienceFunction      = standard::AlgorithmFactory::create("PitchSalienceFunction");
  _pitchSalienceFunctionPeaks = standard::AlgorithmFactory::create("PitchSalienceFunctionPeaks");
  _pitchContours              = standard::AlgorithmFactory::create("PitchContours");

  declareInput(_signal, 4096, "signal", "the input audio signal");
  declareOutput(_rMatrix, "r", "the quantile ratios matrix");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_saliences, "saliences", "humming tones saliences");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  // Connect input proxy.
  _signal >> _decimator->input("signal");

  _decimator->output("signal").setBufferType(BufferUsage::forLargeAudioStream);

  _decimator->output("signal") >> _lowPass->input("signal");

  _lowPass->output("signal").setBufferType(BufferUsage::forLargeAudioStream);

  _lowPass->output("signal") >> _frameCutter->input("signal");

  _frameCutter->output("frame") >> _welch->input("frame");

  _welch->output("psd") >> PC(_pool, "psd");

  _network = new scheduler::Network(_decimator);
}

HumDetector::~HumDetector() {
  delete _network;
}


void HumDetector::configure() {
  _outSampleRate = 2000.f;
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = int(round(parameter("hopSize").toReal() * _outSampleRate));
  uint frameSize = int(round(parameter("frameSize").toReal() * _outSampleRate));
  _frameSize = nextPowerTwo(frameSize);
  _timeWindow = int(round(parameter("timeWindow").toReal() * _outSampleRate / _hopSize));
  _Q0 = parameter("Q0").toReal();
  _Q1 = parameter("Q1").toReal();
  _maximumFrequency = parameter("maximumFrequency").toReal();
  _minDuration = parameter("minimumDuration").toReal() * 1000.f;
  _timeContinuity = parameter("timeContinuity").toReal() * 1000.f;
  _minimumFrequency = parameter("minimumFrequency").toReal();
  _numberHarmonics = parameter("numberHarmonics").toInt();
  _detectionThreshold = parameter("detectionThreshold").toReal();

  _medianFilterSize = _frameSize * 60 / (_outSampleRate);
  _medianFilterSize += (_medianFilterSize + 1) % 2;

  _decimator->configure("inputSampleRate", _sampleRate,
                        "outputSampleRate", _outSampleRate,
                        "quality", 2);

  // Resample buffers have to be resized everytime it is configured 
  // as it resets the sizes to the default each time.
  _decimator->output("signal").setBufferType(BufferUsage::forLargeAudioStream);

  _lowPass->configure("sampleRate",_outSampleRate,
                      "cutoffFrequency", _maximumFrequency);

  _frameCutter->configure("frameSize",frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "keep");

  _welch->configure("fftSize",_frameSize,
                    "frameSize", frameSize,
                    "averagingFrames", 2,
                    "windowType", "blackmanharris92");

  _spectralPeaks->configure("sampleRate", _outSampleRate,
                            "minFrequency", _minimumFrequency,
                            "maxFrequency", _maximumFrequency,
                            "magnitudeThreshold", _detectionThreshold,
                            "maxPeaks", 5);

  Real binResolution = 20;
  _binsInOctave = 1200.0 / binResolution;
  _pitchSalienceFunction->configure("binResolution", binResolution,
                                    "harmonicWeight", 0.1,
                                    "referenceFrequency", _minimumFrequency,
                                    "numberHarmonics", _numberHarmonics);

  _pitchSalienceFunctionPeaks->configure("binResolution", binResolution,
                                         "maxFrequency", _maximumFrequency,
                                         "minFrequency", _minimumFrequency,
                                         "referenceFrequency",  _minimumFrequency);

  Real binsToSkip = 6;

  Real pitchContinuity = (binsToSkip / _binsInOctave) * 1200. / (1000.f * _hopSize / _outSampleRate); 
  _pitchContours->configure("binResolution", binResolution, 
                            "hopSize", _hopSize,
                            "sampleRate", _outSampleRate,
                            "minDuration", _minDuration,
                            "pitchContinuity", pitchContinuity,
                            "timeContinuity", _timeContinuity);


  _referenceTerm = 0.5 - _binsInOctave * log2(_minimumFrequency);

  _EPS = numeric_limits<Real>::epsilon();
}


AlgorithmStatus HumDetector::process() {
  if (!shouldStop()) return PASS;

  if (!_pool.contains<vector<vector<Real> > >("psd")) {
    // Do not push anything in the case of empty signal.
    E_WARNING("HumDetector: empty input signal");
    return FINISHED;
  }

  const vector<vector<Real> >& psd = _pool.value<vector<vector<Real> > >("psd");

  _timeStamps = psd.size();
  _spectSize = psd[0].size();

  if (_timeStamps < 10) {
    E_INFO("HumDetector: With only " << _timeStamps << 
           " PSD frames it is not posible to estimate humming frequencies."
           " Try to process a longer audio stream or to reduce the hopSize parameter");

    _rMatrix.push(TNT::Array2D<Real>());
    _frequencies.push(vector<Real>());
    _saliences.push(vector<Real>());
    _starts.push(vector<Real>());
    _ends.push(vector<Real>());

    return FINISHED;
  }

  // This algorithm relies in a long audio window (10s by default). In order to prevent it to break,
  // the analysis window duration shrinks down when the input audio stream is shorter.
  if (_timeWindow > _timeStamps) {
    E_INFO("HumDetector: the selected time window needs " << _timeWindow * _hopSize / _outSampleRate << 
    "s of audio while the input stream lasts " << _timeStamps * _hopSize / _outSampleRate << 
    "s.  Resizing the analysis time window to " << _timeStamps * _hopSize / (2 * _outSampleRate) << "s");
    _timeWindow = _timeStamps/2;
  }

  // The algorithms works by sorting (energy-wise) the PSD of the analysis window and computing the 
  // ratio between the Q0 and Q1 quantiles (0.1 and 0.55 by default).
  _Q0sample = (uint)(_Q0 * _timeWindow + 0.5);
  _Q1sample = (uint)(_Q1 * _timeWindow + 0.5);

  _iterations = _timeStamps - _timeWindow + 1;
  vector<vector<Real> > psdWindow(_spectSize, vector<Real>(_timeWindow, 0.f));
  vector<vector<Real> > r(_spectSize, vector<Real>(_iterations, 0.f));
  vector<size_t> psdIdxs(_timeWindow, 0);
  Real Q0, Q1;

  // Initialize the PSD and r (quantile ratios) matrices.
  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = 0; j < _timeWindow; j++)
      psdWindow[i][j] = psd[j][i];

    psdIdxs = sort_indexes(psdWindow[i]);
    Q0 = psdWindow[i][psdIdxs[_Q0sample]];
    Q1 = psdWindow[i][psdIdxs[_Q1sample]];
    
    r[i][0] = Q0 / (Q1 + _EPS);
  }

  // Iterate during the remaining timestamps.
  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = _timeWindow; j < _timeStamps; j++) {
      rotate(psdWindow[i].begin(), psdWindow[i].begin() + 1, psdWindow[i].end());
      psdWindow[i][_timeWindow - 1] =psd[j][i];
      psdIdxs = sort_indexes(psdWindow[i]);

      Q0 = psdWindow[i][psdIdxs[_Q0sample]];
      Q1 = psdWindow[i][psdIdxs[_Q1sample]];

      r[i][j - _timeWindow + 1] = Q0 / (Q1 + _EPS);
      }
  }

  // Apply the median filter frequency-wise.
  vector<Real> rSpec = vector<Real>(_spectSize, 0.f);
  vector<Real> filtered = vector<Real>(_spectSize, 0.f);
  _Smoothing->configure("kernelSize", _medianFilterSize); 
  _Smoothing->output("filteredArray").set(filtered);
  _Smoothing->input("array").set(rSpec);
  for (uint j = 0; j < _iterations; j++) {
    for (uint i = 0; i < _spectSize; i++) {
      rSpec[i] = r[i][j];
    }
    _Smoothing->compute();
    
    for (uint i = 0; i < _spectSize; i++)
      r[i][j] -= filtered[i];
  }

  // Apply the median filter time-wise.
  uint kernerSize = min((uint)(_timeWindow / 2), _iterations);
  kernerSize -= (kernerSize + 1) % 2;
  _Smoothing->configure("kernelSize", kernerSize);
  _Smoothing->output("filteredArray").set(filtered);

  for (uint i = 0; i < _spectSize; i++) {
    _Smoothing->input("array").set(r[i]);
    _Smoothing->compute();

    for (uint j = 0; j < _iterations; j++)
      r[i][j] = filtered[j];
  }

  _rMatrix.push(vecvecToArray2D(r));
  
  vector<Real> frequencies, magnitudes;
  vector<Real> salienceFunction;
  vector<Real> salienceBins, salienceValues;
  vector<vector<Real> >peakBins(_iterations);
  vector<vector<Real> >peakSaliences(_iterations);
  bool peakBinsNotEmpty = false;
  Real threshold;

  // finally the r matrix is feed into the pitch contours recommended signal chain
  for (uint j = 0; j < _iterations; j++) {
    for (uint i = 0; i < _spectSize; i++) 
      rSpec[i] = r[i][j];

    threshold = _detectionThreshold * stddev(rSpec, mean(rSpec));

    _spectralPeaks->configure("sampleRate", _outSampleRate,
                              "minFrequency", _minimumFrequency,
                              "maxFrequency", _maximumFrequency,
                              "magnitudeThreshold", threshold,
                              "maxPeaks", 5);

    _spectralPeaks->input("spectrum").set(rSpec);
    _spectralPeaks->output("frequencies").set(frequencies);
    _spectralPeaks->output("magnitudes").set(magnitudes);
    _spectralPeaks->compute();

    _pitchSalienceFunction->input("frequencies").set(frequencies);
    _pitchSalienceFunction->input("magnitudes").set(magnitudes);
    _pitchSalienceFunction->output("salienceFunction").set(salienceFunction);
    _pitchSalienceFunction->compute();

    _pitchSalienceFunctionPeaks->input("salienceFunction").set(salienceFunction);
    _pitchSalienceFunctionPeaks->output("salienceBins").set(salienceBins);
    _pitchSalienceFunctionPeaks->output("salienceValues").set(salienceValues);
    _pitchSalienceFunctionPeaks->compute();

    peakBins[j] = salienceBins;
    peakSaliences[j] = salienceValues;

    if (not salienceBins.empty())
      peakBinsNotEmpty = true;
  }

  std::vector<std::vector<Real> > contoursBins;
  std::vector<std::vector<Real> > contoursSaliences;
  std::vector<Real> contoursStartTimes, contoursEndsTimes, contoursFreqsMean, contoursSaliencesMean;
  Real duration;

  if (peakBinsNotEmpty) {
    _pitchContours->input("peakBins").set(peakBins);
    _pitchContours->input("peakSaliences").set(peakSaliences);
    _pitchContours->output("contoursBins").set(contoursBins);
    _pitchContours->output("contoursSaliences").set(contoursSaliences);
    _pitchContours->output("contoursStartTimes").set(contoursStartTimes);
    _pitchContours->output("duration").set(duration);
    _pitchContours->compute();

    contoursFreqsMean.assign(contoursBins.size(), 0.f);
    contoursSaliencesMean.assign(contoursBins.size(), 0.f);
    contoursEndsTimes.assign(contoursBins.size(), 0.f);

    Real timeWindowSecs = _timeWindow * _hopSize / _outSampleRate;
    for (uint i = 0; i < contoursBins.size(); i++) {
      // We add the offset due to the initial frames needed to fill the buffers
      // to the starts and ends.
      contoursStartTimes[i] += timeWindowSecs;
      contoursFreqsMean[i] = centBinToFrequency(mean(contoursBins[i]), 
                                                _referenceTerm, _binsInOctave);
      contoursSaliencesMean[i] = mean(contoursSaliences[i]);
      contoursEndsTimes[i] = contoursStartTimes[i] + 
                              contoursSaliences[i].size() * _hopSize / _outSampleRate;
    }
  }

  _frequencies.push(contoursFreqsMean);
  _saliences.push(contoursSaliencesMean);
  _starts.push(contoursStartTimes);
  _ends.push(contoursEndsTimes);

  return FINISHED;
}


void HumDetector::reset() {
  AlgorithmComposite::reset();
  _decimator->output("signal").setBufferType(BufferUsage::forLargeAudioStream);
  _pool.remove("psd");
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* HumDetector::name = "HumDetector";
const char* HumDetector::category = "Audio Problems";
const char* HumDetector::description = DOC("This algorithm detects low frequency tonal noises in the audio signal. "
"First, the steadiness of the Power Spectral Density (PSD) of the signal is computed by measuring the quantile ratios "
"as described in [1]. After this, the PitchContours algorithm is used to keep track of the humming tones [2].\n"
"\n"
"References:\n"
"  [1] Brandt, M., & Bitzer, J. (2014). Automatic Detection of Hum in Audio\n"
"  Signals. Journal of the Audio Engineering Society, 62(9), 584-595.\n"
"  [2] J. Salamon and E. Gómez, Melody extraction from polyphonic music signals\n"
"  using pitch contour characteristics, IEEE Transactions on Audio, Speech,\n"
"  and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n");


HumDetector::HumDetector() {
  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_rMatrix, "r", "the quantile ratios matrix");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_saliences, "saliences", "humming tones saliences");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  createInnerNetwork();
}


HumDetector::~HumDetector() {
  delete _network;
}


void HumDetector::configure() {
  _humDetector->configure(INHERIT("sampleRate"), INHERIT("hopSize"),
                          INHERIT("frameSize"), INHERIT("timeWindow"),
                          INHERIT("minimumFrequency"), INHERIT("maximumFrequency"),
                          INHERIT("Q0"), INHERIT("Q1"),
                          INHERIT("minimumDuration"), INHERIT("timeContinuity"),
                          INHERIT("numberHarmonics"), INHERIT("detectionThreshold"));
}


void HumDetector::createInnerNetwork() {
  _humDetector = streaming::AlgorithmFactory::create("HumDetector");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _humDetector->input("signal");
  _humDetector->output("r")    >> PC(_pool, "r");
  _humDetector->output("frequencies")    >> PC(_pool, "frequencies");
  _humDetector->output("saliences")    >> PC(_pool, "saliences");
  _humDetector->output("starts")   >> PC(_pool, "starts");
  _humDetector->output("ends")   >> PC(_pool, "ends");

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
  vector<Real>& amplitudes = _saliences.get();
  vector<Real>& starts = _starts.get();
  vector<Real>& ends = _ends.get();

  rMatrix = _pool.value<vector<TNT::Array2D<Real> > >("r")[0];
  frequencies = _pool.value<vector<Real> >("frequencies");
  amplitudes = _pool.value<vector<Real> >("saliences");
  starts = _pool.value<vector<Real> >("starts");
  ends = _pool.value<vector<Real> >("ends");

  reset();
}


void HumDetector::reset() {
  _network->reset();
  _pool.remove("r");
  _pool.remove("frequencies");
  _pool.remove("saliences");
  _pool.remove("starts");
  _pool.remove("ends");
}

} // namespace standard
} // namespace essentia
