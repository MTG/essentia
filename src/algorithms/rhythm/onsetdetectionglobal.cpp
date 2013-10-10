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

#include "onsetdetectionglobal.h"
#include <complex>
#include <limits>
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {


const char* OnsetDetectionGlobal::name = "OnsetDetectionGlobal";
const char* OnsetDetectionGlobal::description = DOC("This algorithm outputs an onset detection function useful for describing onset occurrences. Detection values are computed frame-wisely given an input signal. The output of this algorithm should be post-processed in order to determine whether the frame contains an onset or not. Namely, it could be fed to the Onsets algorithm.\n"
"The following method are available:\n"
"  - 'infogain', the spectral difference measured by the modified information gain [1]. For each frame, it accounts for energy change in between preceding and consecutive frames, histogrammed together, in order to suppress short-term variations on frame-by-frame basis.\n"
"  - 'beat_emphasis', the beat emphasis function [1]. This function is a linear combination of onset detection functions (complex spectral differences) in a number of sub-bands, weighted by their beat strength computed over the entire input signal."
"\n"
"Note:\n"
"  - 'infogain' onset detection has been optimized for the default sampleRate=44100Hz, frameSize=2048, hopSize=512.\n"
"  - 'beat_emphasis' is optimized for a fixed resolution of 11.6ms, which corresponds to the default sampleRate=44100Hz, frameSize=1024, hopSize=512.\n"
"  Optimal performance of beat detection with TempoTapDegara is not guaranteed for other settings.\n"
"\n"
"References:\n"
"  [1] S. Hainsworth and M. Macleod, \"Onset detection in musical audio\n"
"  signals,\" in International Computer Music Conference (ICMC’03), 2003,\n"
"  pp. 163–6.\n\n"
"  [2] M. E. P. Davies, M. D. Plumbley, and D. Eck, \"Towards a musical beat\n"
"  emphasis function,\" in IEEE Workshop on Applications of Signal Processing\n"
"  to Audio and Acoustics, 2009. WASPAA  ’09, 2009, pp. 61–64.");


void OnsetDetectionGlobal::configure() {
  Real sampleRate = parameter("sampleRate").toReal();
  _method = parameter("method").toLower();
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();

  // Frames are cut starting from zero as in the paper and consistently with
  // OnsetRate algorithm
  _frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "startFromZero", true);

  _windowing->configure("size", frameSize,
                        "zeroPadding", 0,
                        "type", "hann");

  if (_method=="infogain") {
    _spectrum->configure("size", frameSize);

    _histogramSize = 5; // use +/- 5 frames around a target frame for histogramming
    _bufferSize = _histogramSize * 2 + 1;

    // reversed triangle weighting
    _weights.clear();
    for (int i=0; i < _histogramSize; i++) {
      Real weight = 1 - i * 0.9/_histogramSize;
      _weights.push_back(weight);                   // 1, 0.82, 0.64, 0.46, 0.28
      _rweights.insert(_rweights.begin(), weight);  // 0.28, 0.46, 0.64, 0.82, 1
    }

    // frequency interval to consider
    Real minFrequency = 40.;
    Real maxFrequency = 5000.;

    // associated FFT bins
    _minFrequencyBin = round(minFrequency * frameSize / sampleRate);
    _maxFrequencyBin = round(maxFrequency * frameSize / sampleRate) + 1;
    _numberFFTBins = _maxFrequencyBin - _minFrequencyBin;
  }

  else if (_method=="beat_emphasis") {
    _numberERBBands = 40;
    _numberFFTBins = int(frameSize)/2 + 1;
    _phase_1.resize(_numberFFTBins);
    _phase_2.resize(_numberFFTBins);
    _spectrum_1.resize(_numberFFTBins);

    _fft->configure("size", frameSize);
    _erbbands->configure("inputSize", frameSize/2 + 1,
                         "numberBands", _numberERBBands,
                         "lowFrequencyBound", 80.,
                         "highFrequencyBound", sampleRate/2,
                         "type", "magnitude");
    // TODO Smoothing window size is set to 8+8 ODF samples as in the paper and
    // matlab code. However, this will result in different time durations for
    // different ODF frame rates. Is a constant time duration required instead?
    // Constant time for default ODF resolution of 11.6ms:
    // Real smoothingTime = 0.18575963718820862;
    // ~0.09s advance + ~0.09s delay;  use a simpler value of 0.2?
    //_smoothingWindowHalfSize = floor(smoothingTime/2 * sampleRate);
    _movingAverage->configure("size", _smoothingWindowHalfSize * 2 + 1);
    _autocorrelation->configure("normalization", "unbiased");

    // Tempo preference weights (Rayleigh distribution, code from TempoTapDegara)
    // Maximum period of ODF to consider (period of 512 ODF samples with the
    // default settings correspond to 512 * 512. / 44100. = ~6 secs
    _maxPeriodODF = int(round(5.944308390022676 * sampleRate / hopSize));
    _weights.resize(_maxPeriodODF);
    Real rayparam2 = pow(round(43 * 512.0/_maxPeriodODF), 2);
    // Rayleigh distribution parameter which sets the strongest point of the weighting
    for (int i=0; i<_maxPeriodODF; ++i) {
      int tau = i+1;
      _weights[i] = tau / rayparam2 * exp(-0.5 * tau*tau / rayparam2);
    }
  }
}

void OnsetDetectionGlobal::compute() {
  const vector<Real>& signal = _signal.get();
  if (signal.empty()) {
    vector<Real>& onsetDetections = _onsetDetections.get();
    onsetDetections.clear();
    return;
  }

  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(_frame);

  _windowing->input("frame").set(_frame);
  _windowing->output("frame").set(_frameWindowed);

  if (_method=="infogain") {
    computeInfoGain();
  }
  else if (_method=="beat_emphasis") {
    computeBeatEmphasis();
  }
}

void OnsetDetectionGlobal::computeInfoGain() {
  vector<Real>& onsetDetections = _onsetDetections.get();

  vector<vector<Real> > buffer(_bufferSize, vector<Real> (_numberFFTBins, 0));
  vector<Real> histogramOld(_numberFFTBins, 0);
  vector<Real> histogramNew(_numberFFTBins, 0);

  vector<Real> spectrum;
  _spectrum->input("frame").set(_frameWindowed);
  _spectrum->output("spectrum").set(spectrum);

  while (true) {
    // get a frame
    _frameCutter->compute();

    if (!_frame.size()) {
      break;
    }
    _windowing->compute();
    _spectrum->compute();

    // update buffer; take only bins we are interested in
    buffer.erase(buffer.begin());
    buffer.push_back(vector<Real>(spectrum.begin() + _minFrequencyBin,
                        spectrum.begin() + _maxFrequencyBin));

    // compute weighted sum of magnitudes for each bin
    for (int b=0; b<_numberFFTBins; b++) {
      // initialize bin
      histogramOld[b] = 0;
      histogramNew[b] = 0;
      for (int i=0; i<_histogramSize; i++) {
        // previous frames
        histogramOld[b] += buffer[i][b] * _rweights[i];
        // posterior frames
        histogramNew[b] += buffer[_histogramSize + 1 + i][b] * _weights[i];
      }
    }

    // Reassign bins with zero magnitude in histogramOld to 1 to avoid division
    // by zero (TODO why to 1?). Reassign bins with zero magnitude in
    // histogramNew to a very little value to avoid log(0)
    /*
      original code by Matthew Davies:
      ind = find(hist1 == 0);
      hist1(ind) = 1;
      if hist1 == 0,  hist1 = 1; end
      if hist2 == 0,  hist2 = eps; end
    */

    Real detection = 0.;
    for (int b=0; b<_numberFFTBins; b++)  {
      if (histogramOld[b] == 0) {
        histogramOld[b] = 1;
      }
      if (histogramNew[b] == 0) {
        histogramNew[b] = numeric_limits<Real>::epsilon();
      }
      // Use information gain as a distance between histogrammed bins in
      // previous and posterior frames. Consider only positive changes.
      detection += max(log2(histogramNew[b] / histogramOld[b]), Real(0));
    }
    onsetDetections.push_back(detection);
  }

  // original algorithm includes smoothing Hanning filter (length = 20 frames)
  // df2 = filtfilt(hanning(20)/sum(hanning(20)),1,df)
  // we omit smoothing, as it should be done on the post-processing stage
}


void OnsetDetectionGlobal::computeBeatEmphasis() {
  vector<Real>& onsetDetections = _onsetDetections.get();
  onsetDetections.clear();

  vector<complex<Real> > frameFFT;
  _fft->input("frame").set(_frameWindowed);
  _fft->output("fft").set(frameFFT);

  vector<Real> spectrum;
  vector<Real> phase;
  _cartesian2polar->input("complex").set(frameFFT);
  _cartesian2polar->output("magnitude").set(spectrum);
  _cartesian2polar->output("phase").set(phase);


  fill(_phase_1.begin(), _phase_1.end(), Real(0.0));
  fill(_phase_2.begin(), _phase_2.end(), Real(0.0));
  fill(_spectrum_1.begin(), _spectrum_1.end(), Real(0.0));

  vector<vector<Real> > onsetERB(_numberERBBands);
  vector<Real> tempFFT (_numberFFTBins, 0.);  // detection function in FFT bins
  vector<Real> tempERB (_numberERBBands, 0.); // detection function in ERP bands

  // NB: a hack to make use of ERBBands algorithm and not reimplement the
  // computation of gammatone filterbank weights again. As long as ERBBands
  // computes weighted magnitudes in each ERB band instead of energy, we can
  // feed it onset detection values instead of spectrum.
  _erbbands->input("spectrum").set(tempFFT);
  _erbbands->output("bands").set(tempERB);

  size_t numberFrames=0;

  while (true) {
    // get a frame
    _frameCutter->compute();

    if (!_frame.size()) {
      break;
    }

    _windowing->compute();
    _fft->compute();
    _cartesian2polar->compute();

    // Compute complex spectral difference. Optimized, see details in the
    // OnsetDetection algo
    for (int i=0; i<_numberFFTBins; ++i) {
      Real targetPhase = 2*_phase_1[i] + _phase_2[i];
      targetPhase = fmod(targetPhase + M_PI, -2 * M_PI) + M_PI;
      tempFFT[i] = norm(_spectrum_1[i] - polar(spectrum[i], phase[i]-targetPhase));
    }

    // Group detection functions for spectral bins into larger ERB sub-bands using
    // a Gammatone filterbank to improve the likelihood of finding meaningful
    // periodicity in spectral bands.
    _erbbands->compute();
    for (int b=0; b<_numberERBBands; ++b) {
      onsetERB[b].push_back(tempERB[b]);
    }

    _phase_2 = _phase_1;
    _phase_1 = phase;
    _spectrum_1 = spectrum;
    numberFrames += 1;
  }

  // Post-processing found in M.Davies' matlab code, but not mentioned in the
  // paper, and skipped in this implementation:
  // - interpolate detection functions by factor of 2 (by zero-stuffing)
  // - half-rectify
  // - apply a Butterworth low-pass filter with zero-phase (running in forward
  // and backward directions); Matlab: [b,a]=butter(2,0.4);
  // - half-rectify again


  if (!numberFrames) {
    return;
  }

  for (int b=0; b<_numberERBBands; ++b) {
    // TODO tmp = interp(newspec2(pp,:),2);
    // interpolate to the doubled sampling rate interp performs lowpass
    // interpolation by inserting zeros into the original sequence and then
    // applying a special lowpass filter.

    // TODO half-rectify is not in the paper, futhermore, all onsetERB values
    // are supposed to be non-negative, as they are weighted sums of norms.
    // Half-rectification would have been necessary in the case of
    // interpolation, which can produce negative values.
    //for (size_t i=0; i<onsetERB[b].size(); ++i) {
    //  if (onsetERB[b][i] < 0) {
    //    onsetERB[b][i] = 0.;
    //  }
    //}

    // TODO newspecout(pp,:) = max(0,(filtfilt(b,a,(tmp))));
    // --> apply lowpass Butterworth filter, half-rectify again

    // normalize to have a unit variance
    Real bandMean = mean(onsetERB[b]);
    Real bandStddev = stddev(onsetERB[b], bandMean);
    if (bandStddev > 0) {
      for (size_t i=0; i<onsetERB[b].size(); ++i) {
        onsetERB[b][i] /= bandStddev;
      }
    }
  }

  // TODO Matlab: sbdb = max(0,newspecout); // half-rectify again? but onsetERB is
  // already non-negative

  // Compute weights for ODFs for ERB bands

  vector<Real> smoothed;
  vector<Real> tempACF;
  vector<vector <Real> > bandsACF;
  bandsACF.resize(_numberERBBands);
  vector <Real> weightsERB;
  weightsERB.resize(_numberERBBands);

  for (int b=0; b<_numberERBBands; ++b) {
    // Apply adaptive moving average threshold to emphasise the strongest and
    // discard the least significant peaks. Subtract the adaptive mean, and
    // half-wave rectify the output, setting any negative valued elements to zero.

    // Align filter output for symmetrical averaging, and we want the filter to
    // return values on the edges as the averager output computed at these
    // positions to avoid smoothing to zero.

    onsetERB[b].insert(onsetERB[b].end(), _smoothingWindowHalfSize, onsetERB[b].back());
    onsetERB[b].insert(onsetERB[b].end(), _smoothingWindowHalfSize, onsetERB[b].back());

    _movingAverage->input("signal").set(onsetERB[b]);
    _movingAverage->output("signal").set(smoothed);
    _movingAverage->compute();

    smoothed.erase(smoothed.begin(), smoothed.begin() + 2*_smoothingWindowHalfSize);
    for (size_t i=0; i<numberFrames; ++i) {
      onsetERB[b][i] -= smoothed[i];
      if (onsetERB[b][i] < 0) {
        onsetERB[b][i] = 0;
      }
    }

    // Compute band-wise unbiased autocorrelation
    _autocorrelation->input("array").set(onsetERB[b]);
    _autocorrelation->output("autoCorrelation").set(tempACF);
    _autocorrelation->compute();

    // Consider only periods up to _maxPeriodODF ODF samples
    tempACF.resize(_maxPeriodODF);

    // Weighten by tempo preference curve
    vector<Real> tempACFWeighted;
    tempACFWeighted.resize(_maxPeriodODF);

    // Apply comb-filtering to reflect periodicities on different metric levels
    // (integer multiples) and apply tempo preference curve.
    int numberCombs = 4;

    // To accout for poor resolution of ACF at short lags, each comb element has
    // width proportional to its relationship to the underlying periodicity, and
    // its height is normalized by its width.

    // 0-th element in autocorrelation vector corresponds to the period of 1.
    // Min value for the 'region' variable is -3 => compute starting from the
    // 3-rd index, which corresponds to the period of 4, until period of 120
    // ODF samples (as in matlab code) or 110 (as in the paper). Generalization:
    // not clear why max period is 120 or 110, should be (512 - 3) / 4 = 127
    int periodMin = 4 - 1;
    int periodMax = (_maxPeriodODF-(numberCombs-1)) / numberCombs - 1;

    for (int comb=1; comb<=numberCombs; ++comb) {
      int width = 2*comb - 1;
      for (int region=1-comb; region<=comb-1; ++region) {
        for (int period=periodMin; period<periodMax; ++period) {
          tempACFWeighted[period] +=
              _weights[period] * tempACF[period*comb + region] / width;
        }
      }
    }

    // We are not interested in the period estimation, but in general salience of
    // the existing periodicity
    weightsERB[b] = tempACFWeighted[argmax(tempACFWeighted)];
  }
  normalize(weightsERB);

  // Matlab M.Davies: take top 40% of weights, zero the rest (not in the paper!)
  vector<Real> sorted;
  sorted.reserve(_numberERBBands);
  copy(weightsERB.begin(), weightsERB.end(), sorted.begin());
  sort(sorted.begin(), sorted.end());
  Real threshold = sorted[int(floor(_numberERBBands * 0.6))];

  // Compute weighted sub of ODFs for ERB bands for each audio frame
  onsetDetections.resize(numberFrames);
  for (size_t i=0; i<numberFrames; ++i) {
    for (int b=0; b<_numberERBBands; ++b) {
      if (weightsERB[b] >= threshold) {
        onsetDetections[i] += onsetERB[b][i] * weightsERB[b];
      }
    }
  }
}


void OnsetDetectionGlobal::reset() {
  Algorithm::reset();
  if (_frameCutter) _frameCutter->reset();
  if (_windowing) _windowing->reset();
  if (_spectrum) _spectrum->reset();
  if (_fft) _fft->reset();
  if (_cartesian2polar) _cartesian2polar->reset();
  if (_movingAverage) _movingAverage->reset();
  if (_erbbands) _erbbands->reset();
  if (_autocorrelation) _autocorrelation->reset();
}


// TODO in the case of lower accuracy in evaluation
// implement post-processing steps for methods in OnsetDetection, which required it
// wrapping the OnsetDetection algo
// - smoothing?
// - etc., whatever was requiered in original matlab implementations

} // namespace standard
} // namespace essentia


#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* OnsetDetectionGlobal::name = standard::OnsetDetectionGlobal::name;
const char* OnsetDetectionGlobal::description = standard::OnsetDetectionGlobal::description;

OnsetDetectionGlobal::OnsetDetectionGlobal() : AlgorithmComposite() {

  _onsetDetectionGlobal = standard::AlgorithmFactory::create("OnsetDetectionGlobal");
  _poolStorage = new PoolStorage<Real>(&_pool, "internal.signal");

  declareInput(_signal, 1, "signal", "the input signal");   // 1
  declareOutput(_onsetDetections, 0, "onsetDetections", "the frame-wise values of the detection function"); // 0

  _signal >> _poolStorage->input("data"); // attach input proxy

  // NB: We want to have the same output stream type as in OnsetDetection for
  // consistency. We need to increase buffer size of the output because the
  // algorithm works on the level of entire track and we need to push all values
  // in the output source at once.
  _onsetDetections.setBufferType(BufferUsage::forLargeAudioStream);
}

OnsetDetectionGlobal::~OnsetDetectionGlobal() {
  delete _onsetDetectionGlobal;
  delete _poolStorage;
}

void OnsetDetectionGlobal::reset() {
  AlgorithmComposite::reset();
  _onsetDetectionGlobal->reset();
}

AlgorithmStatus OnsetDetectionGlobal::process() {
  if (!shouldStop()) return PASS;

  vector<Real> detections;
  //const vector<Real>& signal = _pool.value<vector<Real> >("internal.signal");

  _onsetDetectionGlobal->input("signal").set(_pool.value<vector<Real> >("internal.signal"));
  _onsetDetectionGlobal->output("onsetDetections").set(detections);
  _onsetDetectionGlobal->compute();

  for(size_t i=0; i<detections.size(); ++i) {
    _onsetDetections.push(detections[i]);
  }
  return FINISHED;
}

} // namespace streaming
} // namespace essentia
