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

#include "tempotapdegara.h"
#include "essentiamath.h"
#include <limits>

using namespace std;

namespace essentia {
namespace standard {


const char* TempoTapDegara::name = "TempoTapDegara";
const char* TempoTapDegara::description = DOC("This algorithm estimates beat positions given an onset detection function.  The detection function is partitioned into 6-second frames with a 1.5-second increment, and the autocorrelation is computed for each frame, and is weighted by a tempo preference curve [2]. Periodicity estimations are done frame-wisely, searching for the best match with the Viterbi algorith [3]. The estimated periods are then passed to the probabilistic beat tracking algorithm [1], which computes beat positions.\n"
"\n"
"Note that the input values of the onset detection functions must be non-negative otherwise an exception is thrown. Parameter \"maxTempo\" should be 20bpm larger than \"minTempo\", otherwise an exception is thrown.\n"
"\n"
"References:\n"
"  [1] Degara, N., Rua, E. A., Pena, A., Torres-Guijarro, S., Davies, M. E., & Plumbley, M. D. (2012). Reliability-informed beat tracking of musical signals. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1), 290-301.\n"
"  [2] Davies, M. E., & Plumbley, M. D. (2007). Context-dependent beat tracking of musical audio. Audio, Speech, and Language Processing, IEEE Transactions on, 15(3), 1009-1020.\n"
"  [3] Stark, A. M., Davies, M. E., & Plumbley, M. D. (2009, September). Real-time beatsynchronous analysis of musical audio. In 12th International Conference on Digital Audio Effects (DAFx-09), Como, Italy.");


void TempoTapDegara::configure() {
  Real minTempo = parameter("minTempo").toInt();
  Real maxTempo = parameter("maxTempo").toInt();

  if (maxTempo < minTempo + 20) {
    throw EssentiaException("TempoTapDegara: maxTempo should be larger than minTempo + 20");
  }

  if(parameter("resample") == "none") _resample = 1;
  else if (parameter("resample") == "x2") _resample = 2;
  else if (parameter("resample") == "x3") _resample = 3;
  else if (parameter("resample") == "x4") _resample = 4;
  _sampleRateODF = parameter("sampleRateODF").toReal() * _resample;

  // ------- M. Davies --------
  // Use hopsize of 1.5 secs, frame size of 6 secs to cut ODF frames. We want
  // to estimate some period for each frame, therefore, the maximum period we
  // can find is limited to the hopsize and corresponds to 40 BPM. Having a
  // frame size 4 times larger, we can take into account the periodicities on
  // integer multiples, using for this purpose up to 4 comb filters in the bank.
  _hopDurationODF = _frameDurationODF / _numberCombs;
  int frameSizeODF = int(round(_frameDurationODF * _sampleRateODF));
  _hopSizeODF = frameSizeODF / _numberCombs;

  _frameCutter->configure("frameSize", frameSizeODF,
                          "hopSize", _hopSizeODF,
                          "startFromZero", true);

  // Smoothing window size of 0.2s: 0.1s advance + 0.1s delay
  _smoothingWindowHalfSize = floor(0.1 * _sampleRateODF);
  _movingAverage->configure("size", _smoothingWindowHalfSize * 2 + 1);
  _autocorrelation->configure("normalization", "unbiased");

  createTempoPreferenceCurve();

  // 0-th element in autocorrelation vector will corresponds to the period of 1.
  // Min value for the 'region' variable is -3 => we will compute starting from i
  // the 3-rd index, which corresponds to the period of 4, until period of 127 =
  // (512-3) / 4 = 127 ODF samples (or until 120 as in matlab code).
  _periodMinIndex = _numberCombs - 1;
  _periodMaxIndex =  (frameSizeODF-(_numberCombs-1)) / _numberCombs - 1;

  // user-defined min/max periods
  _periodMaxUserIndex = (int)ceil(60. / minTempo * _sampleRateODF) - 1;
  _periodMinUserIndex = (int)floor(60. / maxTempo * _sampleRateODF) - 1;

  // make sure user-specified indexes are within the allowed range
  _periodMinUserIndex = min(_hopSizeODF-1, _periodMinUserIndex);
  _periodMaxUserIndex = min(_hopSizeODF-1, _periodMaxUserIndex);

  // TODO we could further adapt frameSizeODF according to maximum desired period
  // instead of using 6 secs:
  //frameSizeODF = pow(2, ceil(log2(_numberCombs * _periodMaxIndexInSamples * _sampleRateODF)));

  createViterbiTransitionMatrix();

  // ------- N. Degara --------
  _resolutionODF = 1. / _sampleRateODF;
}


void TempoTapDegara::reset() {
   Algorithm::reset();
    if (_movingAverage) _movingAverage->reset();
    if (_frameCutter) _frameCutter->reset();
    if (_autocorrelation) _autocorrelation->reset();
}


void TempoTapDegara::compute() {

  vector<Real> detections = _onsetDetections.get(); // we need a copy
  vector<Real>& ticks = _ticks.get();

  // sanity checks
  for(size_t i=0; i<detections.size(); ++i) {
    if (detections[i]<0) {
      throw EssentiaException("TempoTapDegara: onset detection values must be non-negative");
    }
  }

  ticks.clear();
  if (!detections.size()) {
    return;
  }

  // Normalize detection function by maximum value
  normalize(detections);

  // optional interpolation
  if (_resample > 1 && detections.size()>1) {
    vector<Real> temp((detections.size()-1)*_resample + 1, 0.);
    for (size_t i=0; i<detections.size()-1; ++i) {
      Real delta = (detections[i+1] - detections[i]) / _resample;
      for (int j=0; j<_resample; ++j) {
        temp[i*_resample + j] = detections[i] + delta*j;
      }
    }
    temp.back() = detections.back();
    detections = temp;
  }

  vector<Real> beatPeriods;
  vector<Real> beatEndPositions;

  computeBeatPeriodsDavies(detections, beatPeriods, beatEndPositions);
  computeBeatsDegara(detections, beatPeriods, beatEndPositions, ticks);
}


void TempoTapDegara::computeBeatsDegara(vector <Real>& detections,
                        const vector<Real>& beatPeriods,
                        const vector<Real>& beatEndPositions,
                        vector<Real>& ticks) {

  // Implementation of Degara's beat tracking using a probabilitic framework
  // (Hidden Markov Model). Tempo estimations throughout the track are assumed
  // to be computed from the algorithm by M. Davies.

  // avoid zeros to avoid log(0) error in future
  for(size_t i=0; i<detections.size(); ++i) {
    if (detections[i]==0) {
      detections[i] = numeric_limits<Real>::epsilon();
    }
  }

  // Minimum tempo (i.e., maximum period) to be considered
  Real periodMax = beatPeriods[argmax(beatPeriods)];
  // The number of states of the HMM is determined bt the largest time between
  // beats allowed (periodMax + 3 standard deviations). Compute a list of
  // inter-beat time intervals corresponding to each state (ignore zero period):
  vector<Real> ibi;
  Real ibiMax = periodMax + 3 *_sigma_ibi;

  ibi.reserve(ceil(ibiMax / _resolutionODF));
  for (Real t=_resolutionODF; t<=ibiMax; t+=_resolutionODF) {
    ibi.push_back(t);
  }
  _numberStates = (int) ibi.size();

  // Compute transition matrix from the inter-beat-interval distribution
  // according to the tempo estimates. Transition matrix is unique for each beat
  // period.
  map<Real, vector<vector<Real> > > transitionMatrix;
  vector<Real> gaussian;
  vector<Real> ibiPDF(_numberStates);

  gaussianPDF(gaussian, _sigma_ibi, _resolutionODF, 0.01 / _resample);
  // Scale down to avoid computational errors,
  // * _resolutionODF, as in matlab code, works worse

  for (size_t i=0; i<beatPeriods.size(); ++i) {
    // no need to recompute if we have seen this beat period before
    if (transitionMatrix.count(beatPeriods[i])==0) {
      // Shift gaussian vector to be centered at beatPeriods[i] secs which is
      // equivalent to round(beatPeriods[i] / _resolutionODF) samples.
      int shift = (int) gaussian.size()/2 - round(beatPeriods[i]/_resolutionODF - 1);
      for (int j=0; j<_numberStates; ++j) {
        int j_new = j + shift;
        ibiPDF[j] = j_new < 0 || j_new >= (int) gaussian.size() ? 0 : gaussian[j_new];
      }
      computeHMMTransitionMatrix(ibiPDF, transitionMatrix[beatPeriods[i]]);
    }
  }

  // Compute observation likelihoods for each HMM state
  vector<vector<Real> > biy;   // _numberStates x _numberFramesODF
  biy.reserve(_numberStates);

  // treat ODF as probability, normalize to 0.99 to avoid numerical problems
  _numberFrames = detections.size();
  vector<Real> beatProbability(_numberFrames);
  vector<Real> noBeatProbability(_numberFrames);
  for (size_t i=0; i<_numberFrames; ++i) {
    beatProbability[i] = 0.99 * detections[i];
    noBeatProbability[i] = 1. - beatProbability[i];
    // NB: work in log space to avoid numerical issues
    beatProbability[i] = (1-_alpha) * log(beatProbability[i]);
    noBeatProbability[i] = (1-_alpha) * log(noBeatProbability[i]);
  }

  biy.push_back(beatProbability);
  biy.insert(biy.end(), _numberStates-1, noBeatProbability);

  // Decoding
  vector<int> stateSequence;
  decodeBeats(transitionMatrix, beatPeriods, beatEndPositions, biy, stateSequence);
  for (size_t i=0; i<stateSequence.size(); ++i) {
    if (stateSequence[i] == 0) { // beat detected
      ticks.push_back(i * _resolutionODF);
    }
  }
}

void TempoTapDegara::decodeBeats(map<Real,
                                 vector<vector<Real> > >& transitionMatrix,
                                 const vector<Real>& beatPeriods,
                                 const vector<Real>& beatEndPositions,
                                 const vector<vector<Real> >& biy,
                                 vector<int>& sequenceStates) {
  // Transition probability matrix at the begining of the track
  size_t currentIndex = 0;

  // Best transition information for backtracking
  vector<vector<int> > stateBacktracking(_numberStates, vector<int>(_numberFrames));

  // HMM cost for each state for the current time
  vector<Real> cost(_numberStates, numeric_limits<Real>::max());
  cost[0] = 0;
  vector<Real> costOld = cost;
  vector<Real> diff(_numberStates);

  // Dynamic programming
  for (size_t t=0; t<_numberFrames; ++t) {
    // Evaluate transitions from any state to state event (state 0)

    // Look for the minimum cost
    for (int i=0; i<_numberStates; ++i) {
      diff[i] = costOld[i] - transitionMatrix[beatPeriods[currentIndex]][i][0];
    }
    int bestState = argmin(diff);
    Real bestPath = diff[bestState];

    if (bestPath==numeric_limits<Real>::max()) {
      bestState = -1;
    }

    // Save best transtions information for backtracking
    stateBacktracking[0][t] = bestState;
    // Update cost; the only possible transition is from state to state+1
    cost[0] = - biy[0][t] + bestPath;
    for (int state=1; state<_numberStates; ++state) {
      cost[state] = costOld[state-1]
                    - transitionMatrix[beatPeriods[currentIndex]][state-1][state]
                    - biy[state][t];
      stateBacktracking[state][t] = state-1;
    }

    // Update cost at t-1
    costOld = cost;

    // Find the transition matrix corresponding to next frame
    if (t+1 < _numberFrames) {
      Real currentTime = (t+1) * _resolutionODF;
      for (size_t i=currentIndex+1; i < beatEndPositions.size() &&
                         beatEndPositions[i] <= currentTime; ++i) {
        currentIndex = i;
      }
    }
  }

  // Decide which of the final states is the most probable

  int finalState = argmin(cost);
  // Backtrace through the model
  sequenceStates.resize(_numberFrames);
  sequenceStates.back() = finalState;
  if (_numberFrames >= 2) {
    for (size_t t=_numberFrames-2; ; --t) {
      sequenceStates[t] = stateBacktracking[sequenceStates[t+1]][t+1];
      if (t==0) {
        break;
      }
    }
  }
}

void TempoTapDegara::computeHMMTransitionMatrix(const vector<Real>& ibiPDF,
                                vector<vector<Real> >& transitions) {

  // Fill in with zeros
  transitions.clear();
  transitions.resize(_numberStates);
  for (int i=0; i<_numberStates; ++i) {
    transitions[i].resize(_numberStates);
  }

  // Estimate transition probabilities
  transitions[0][0] = ibiPDF[0];
  transitions[0][1] = 1 - transitions[0][0];
  for (int i=1; i<_numberStates; ++i) {
    vector<Real> temp(i, 0.);
    for (int k=0; k<i; ++k) {
      temp[k] = log(transitions[k][k+1]);
    }
    transitions[i][0] = exp(log(ibiPDF[i]) - sum(temp));

    // Matlab: check for numerical problems (probabilities should be within [0,1])
    if (transitions[i][0] < 0 || transitions[i][0] > 1) {
      cerr << "Numerical problems in TempoTapDegara::computeHMMTransitionMatrix" << endl;
      // TODO should be Essentia exception instead?
      // truncate to 1 to avoid further NaNs in log computation
      if (transitions[i][0] < 0) {
        transitions[i][0] = 0;
      }
      else {
        transitions[i][0] = 1;
      }
    }
    if (i+1 < _numberStates) {
      transitions[i][i+1] = 1 - transitions[i][0];
    }
  }

  // NB: work in log space to avoid numerical issues
  for (int i=0; i<_numberStates; ++i) {
    for (int j=0; j<_numberStates; ++j) {
      transitions[i][j] = log(transitions[i][j]) * _alpha;
    }
  }
}


void TempoTapDegara::computeBeatPeriodsDavies(vector<Real> detections,
                              vector<Real>& beatPeriods,
                              vector<Real>& beatEndPositions) {
  // Implementation of the beat period detection algorithm by M. Davies.

  adaptiveThreshold(detections, _smoothingWindowHalfSize);

  // Tempo estimation:
  // - Split detection function into overlapping frames.
  // - Compute autocorrelation (ACF) for each frame with bias correction.
  // - Weight it by the tempo preference curve (Rayleigh distrubution).

  vector<vector<Real> > observations;
  Real observationsMax = 0;
  vector<Real> frame;
  vector<Real> frameACF;
  vector<Real> frameACFNormalized(_hopSizeODF);

  _frameCutter->input("signal").set(detections);
  _frameCutter->output("frame").set(frame);
  _autocorrelation->input("array").set(frame);
  _autocorrelation->output("autoCorrelation").set(frameACF);

  while (true) {
    // get a frame
    _frameCutter->compute();
    if (!frame.size()) {
      break;
    }
    _autocorrelation->compute();
    // To accout for poor resolution of ACF at short lags, each comb element has
    // width proportional to its relationship to the underlying periodicity, and
    // its height is normalized by its width.
    fill(frameACFNormalized.begin(), frameACFNormalized.end(), (Real)0.0);
    for (int comb=1; comb<=_numberCombs; ++comb) {
      int width = 2*comb - 1;
      for (int region=1-comb; region<=comb-1; ++region) {
        for (int period=_periodMinIndex; period<=_periodMaxIndex; ++period) {
          frameACFNormalized[period] +=
              _tempoWeights[period] * frameACF[(period+1)*comb-1 + region] / width;
        }
      }
    }
    // Apply adaptive threshold. It is not mentioned in the paper, but is taken
    // from matlab code by M.Davies (including the smoothing size). The
    // implemented smoothing does not exactly match the one in matlab code,
    // howeer, the evaluation results were found very close.
    adaptiveThreshold(frameACFNormalized, 8);

    // zero weights for periods out of the user-specified range
    fill(frameACFNormalized.begin(), frameACFNormalized.begin() + _periodMinUserIndex+1, (Real) 0.);
    fill(frameACFNormalized.begin() + _periodMaxUserIndex+1, frameACFNormalized.end(), (Real) 0.);

    normalizeSum(frameACFNormalized);
    observations.push_back(frameACFNormalized);

    // Search for the maximum value in observations in the same loop.
    Real tMax = observations.back()[argmax(observations.back())];
    if (tMax > observationsMax) {
      observationsMax = tMax;
    }
  }
  _frameCutter->reset();  // TODO reset here for consequent signal inputs, or should the user do it always?

  _numberFramesODF = observations.size();
  // Add noise
  for (size_t t=0; t<_numberFramesODF; ++t) {
    for (int i=0; i<_hopSizeODF; ++i) {
      observations[t][i] += 0.0001 * observationsMax * (float) rand() / RAND_MAX;
    }
  }

  // find Viterbi path (ODF-frame-wise list of indices of the estimated periods;
  // zero index corresponds to beat period of 1 ODF frame hopsize)
  vector <Real> path;
  findViterbiPath(_tempoWeights, _transitionsViterbi, observations, path);

  beatPeriods.reserve(_numberFramesODF);
  beatEndPositions.reserve(_numberFramesODF);

  for (size_t t=0; t<_numberFramesODF; ++t) {
    beatPeriods.push_back((path[t]+1) / _sampleRateODF);
    beatEndPositions.push_back((t + 1) * _hopDurationODF);
  }
}


void TempoTapDegara::findViterbiPath(const vector<Real>& prior,
                     const vector<vector<Real> > transitionMatrix,
                     const vector<vector <Real> >& observations,
                     vector<Real>& path) {
  // Find the most-probable (Viterbi) path through the HMM state trellis.

  // Inputs:
  //   prior(i) = Pr(Q(1) = i)
  //   transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
  //   observations(i,t) = Pr(y(t) | Q(t)=i)
  //
  // Outputs:
  //   path(t) = q(t), where q1 ... qT is the argmax of the above expression.

  // delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
  // psi(j,t) = the best predecessor state, given that we ended up in state j at t

  int numberPeriods = prior.size();

  vector<vector<Real> > delta; // = zeros(numberFramesODF,numberPeriods);
  vector<vector<Real> > psi;   // = zeros(numberFramesODF,numberPeriods);

  vector<Real> deltaNew;
  deltaNew.resize(numberPeriods);

  // weighten likelihoods of periods in the first frame by the prior
  for (int i=0; i<numberPeriods; ++i) {
    deltaNew[i] = prior[i] * observations[0][i];
  }
  normalizeSum(deltaNew);
  delta.push_back(deltaNew);

  vector<Real> psiNew;
  // a vector of zeros (arbitrary, since there is no predecessor to the first frame)
  psiNew.resize(numberPeriods);
  psi.push_back(psiNew);

  vector<Real> tmp;
  tmp.resize(numberPeriods);

  for (size_t t=1; t<_numberFramesODF; ++t) {
    for (int j=0; j<numberPeriods; ++j) {
      for (int i=0; i<numberPeriods; ++i) {
        // weighten delta for a previous frame by vector from the transitionMatrix
        tmp[i] = delta.back()[i] * transitionMatrix[j][i];
      }
      int iMax = argmax(tmp);
      deltaNew[j] = tmp[iMax] * observations[t][j];
      psiNew[j] = iMax;
    }
    normalizeSum(deltaNew);
    delta.push_back(deltaNew);
    psi.push_back(psiNew);
  }

  // track the path backwards in time
  path.resize(_numberFramesODF);
  path.back() = argmax(delta.back());
  if (_numberFramesODF >= 2) {
    for (size_t t=_numberFramesODF-2;; --t) {
      path[t] = psi[t+1][path[t+1]];
      if (t==0) { // size_t can't be negative, break on zero
        break;
      }
    }
  }
}


void TempoTapDegara::createViterbiTransitionMatrix() {
  // Prepare a transition matrix for Viterbi algorithm: it is a _hopSizeODF x
  // _hopSizeODF matrix, where each column i consists of a gaussian centered
  // at i, with stddev=8 by default (i.e., when _hopSizeODF=128), and leave
  // columns before 28th and after 108th zeroed, as well as the lines before
  // 28th and after 108th. Paper: informal tests revealed that stddev parameter
  // can vary by a factor of 2 without altering the overall performance of beat
  // tracker.

  // Generalize values to any ODF sample rate.

  _transitionsViterbi.resize(_hopSizeODF);
  for (int i=0; i<_hopSizeODF; ++i) {
    _transitionsViterbi[i].resize(_hopSizeODF);
  }

  Real scale = _sampleRateODF / (44100./512);

  // each sequent column contains a gaussian shifted by 1 line
  vector<Real> gaussian;
  gaussianPDF(gaussian, 8*scale, 1.);

  int minIndex = floor(28 * scale) - 1;  // because 0-th index is 1st column/line
  int maxIndex = ceil(108 * scale) - 1;
  int gaussianMean = gaussian.size() / 2;

  for (int i=minIndex; i<=maxIndex; ++i) {
    // gaussian with mean=i, std=8*scale;
    for (int j=i-gaussianMean; j<=i+gaussianMean; ++j) {
      if (j>=minIndex && j <= maxIndex) {
        _transitionsViterbi[i][j] = gaussian[j - (i-gaussianMean)];
      }
    }
  }
}


void TempoTapDegara::gaussianPDF(vector<Real>& gaussian, Real gaussianStd, Real step, Real scale) {
  // Estimate probability density function on an interval within 3 standard
  // deviations which will cover 99.7% of cases. Central element of gaussian
  // vector contains mean value, +/- 1 index contain PDF for +/- 1 step.
  // Gaussian height is scaled according to the scale parameter.

  int gaussianSize = 2 * ceil(4 * gaussianStd / step) + 1;
  // better precision with +/- 4 std from the mean than with +/- 3 std

  int gaussianMean = gaussianSize / 2;  // index of the Gaussian's mean
  gaussian.resize(gaussianSize);

  // 1 / ( std * sqrt(2*pi) ) * exp( -pow(i-mean,2)/ (2*pow(std,2)) )
  Real term1 = 1. / (gaussianStd * sqrt(2*M_PI));
  Real term2 = -2 * pow(gaussianStd, 2);
  for (int i=0; i<=gaussianMean; ++i) {
    gaussian[i] = term1 * exp(pow((i-gaussianMean)*step, 2) / term2) * scale;
    if (gaussian[i] < 1e-12) {
      gaussian[i] = 0; // null very low numbers to avoid numerical errors
    }
    gaussian[gaussianSize-1 - i] = gaussian[i];
  }
}


void TempoTapDegara::createTempoPreferenceCurve() {
  // Tempo preference weights (Rayleigh distribution) with a peak at 120 BPM,
  // equal to pow(43, 2) with the default ODF sample rate (44100./512).
  // Maximum period of ODF to consider (period of 512 ODF samples with the
  // default settings) correspond to 512 * 512. / 44100. = ~6 secs
  Real rayparam2 = pow(round(60 * _sampleRateODF / 120), 2);
  int _maxPeriod = _hopSizeODF;
  _tempoWeights.resize(_maxPeriod);
  for (int i=0; i<_maxPeriod; ++i) {
    int tau = i+1;
    _tempoWeights[i] = tau / rayparam2 * exp(-0.5 * tau*tau / rayparam2);
  }
  normalizeSum(_tempoWeights);
  _movingAverage->reset();
}


void TempoTapDegara::adaptiveThreshold(vector<Real>& array, int smoothingHalfSize) {
  // Adaptive moving average threshold to emphasize the strongest and discard the
  // least significant peaks. Subtract the adaptive mean, and half-wave rectify
  // the output, setting any negative valued elements to zero.

  // Align filter output for symmetrical averaging, and we want the filter to
  // return values on the edges as the averager output computed at these
  // positions to avoid smoothing to zero.

  array.insert(array.begin(), smoothingHalfSize, array.front());
  array.insert(array.end(), smoothingHalfSize, array.back());
  vector<Real> smoothed;
  _movingAverage->input("signal").set(array);
  _movingAverage->output("signal").set(smoothed);
  _movingAverage->compute();

  array.erase(array.begin(), array.begin() + smoothingHalfSize);
  array.erase(array.end() - smoothingHalfSize, array.end());
  for (size_t i=0; i<array.size(); ++i) {
    array[i] -= smoothed[2*smoothingHalfSize + i];
    if (array[i] < 0) { // half-rectify
      array[i] = 0;
    }
  }
}

} // namespace standard
} // namespace essentia


#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* TempoTapDegara::name = standard::TempoTapDegara::name;
const char* TempoTapDegara::description = standard::TempoTapDegara::description;


TempoTapDegara::TempoTapDegara() : AlgorithmComposite() {

  _tempoTapDegara = standard::AlgorithmFactory::create("TempoTapDegara");
  _poolStorage = new PoolStorage<Real>(&_pool, "internal.detections");

  declareInput(_onsetDetections, 1, "onsetDetections", "per-frame onset detection values");
  declareOutput(_ticks, 0, "ticks", "the list of resulting ticks [s]");

  _onsetDetections >> _poolStorage->input("data"); // attach input proxy

  // NB: We want to have the same output stream type as in TempoTapTicks for
  // consistency. We need to increase buffer size of the output because the
  // algorithm works on the level of entire track and we need to push all values
  // in the output source at once.
  _ticks.setBufferType(BufferUsage::forLargeAudioStream);
}


TempoTapDegara::~TempoTapDegara() {
  delete _tempoTapDegara;
  delete _poolStorage;
}


void TempoTapDegara::reset() {
  AlgorithmComposite::reset();
  _tempoTapDegara->reset();
}


AlgorithmStatus TempoTapDegara::process() {
  if (!shouldStop()) return PASS;

  vector<Real> ticks;
  _tempoTapDegara->input("onsetDetections").set(_pool.value<vector<Real> >("internal.detections"));
  _tempoTapDegara->output("ticks").set(ticks);
  _tempoTapDegara->compute();

  for(size_t i=0; i<ticks.size(); ++i) {
    _ticks.push(ticks[i]);
  }
  return FINISHED;
}

} // namespace streaming
} // namespace essentia

