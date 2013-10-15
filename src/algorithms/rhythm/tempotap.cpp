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

#include "tempotap.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TempoTap::name = "TempoTap";
const char* TempoTap::description = DOC("This algorithm estimates the periods and phases of a periodic signal, represented by a sequence of values of any number of detection functions, such as energy bands, onsets locations, etc. It requires to be sequentially run on a vector of such values (\"featuresFrame\") for each particular audio frame in order to get estimations related to that frames. The estimations are done for each detection function separately, utilizing the latest \"frameHop\" frames, including the present one, to compute autocorrelation. Empty estimations will be returned until enough frames are accumulated in the algorithm's buffer.\n"
"The algorithm uses elements of the following beat-tracking methods:\n"
" - BeatIt, elaborated by Fabien Gouyon and Simon Dixon (input features) [1]\n"
" - Multi-comb filter with Rayleigh weighting, Mathew Davies [2]\n"
"\n"
"Parameter \"maxTempo\" should be 20bpm larger than \"minTempo\", otherwise an exception is thrown. The same applies for parameter \"frameHop\", which should not be greater than numberFrames. If the supplied \"tempoHints\" did not match any realistic bpm value, an exeception is thrown.\n"
"\n"
"This algorithm is thought to provide the input for TempoTapTicks algorithm. The \"featureFrame\" vectors can be formed by Multiplexer algorithm in the case of combining different features.\n"
"\n"
"Quality: outdated (use TempoTapDegara instead)\n"
"\n"
"References:\n"
"  [1] F. Gouyon, \"A computational approach to rhythm description: Audio\n"
"  features for the computation of rhythm periodicity functions and their use\n"
"  in tempo induction and music content processing,\" UPF, Barcelona, Spain,\n"
"  2005.\n\n"
"  [2] M. Davies and M. Plumbley, \"Causal tempo tracking of audio,\" in\n"
"  International Symposium on Music Information Retrieval (ISMIR'04), 2004.");


void TempoTap::reset() {
  // WARNING: MAGIC NUMBER
  int nfeats = 11;
  _featuresOld = vector<vector<Real> >(_numberFrames - _frameHop,
                                       vector<Real>(nfeats, 0.0));
  _featuresNew.clear();
}

void TempoTap::configure() {
  Real minTempo = parameter("minTempo").toInt();
  Real maxTempo = parameter("maxTempo").toInt();

  if (maxTempo < minTempo + 20) {
    throw EssentiaException("maxTempo should be larger than minTempo + 20");
  }

  _numberFrames = parameter("numberFrames").toInt();
  _frameHop = parameter("frameHop").toInt();

  if (_numberFrames < _frameHop) {
    throw EssentiaException("frameHop should be smaller or equal to numberFrames");
  }

  _frameTime = parameter("frameSize").toReal() / parameter("sampleRate").toReal();

  _minLag = (int)floor(60. / _frameTime / maxTempo);
  _maxLag = (int)ceil(60. / _frameTime / minTempo);


  // WARNING: MAGIC NUMBER
  _nPeaks = 4;

  // set the acf normalisation mode
  _autocorr->configure("normalization", "unbiased");

  // create rayleigh weighting vector applied to comb filter
  _maxelem = 4;
  _comblen = _numberFrames  / _maxelem;
  _weighting.resize(_comblen);
  // WARNING: MAGIC NUMBER
  Real rayParam = 48./512. * _numberFrames;
  Real rayParam2 = rayParam * rayParam;

  for (int i=1; i<=(int)_weighting.size(); i++) {
    _weighting[i-1] = i / rayParam2 * exp(-i*i / (2*rayParam2));
  }


  // if the user provided some tempo hints, use these to have a custom weighting
  // function instead of the Rayleigh one we would use normally
  vector<Real> tempoHints = parameter("tempoHints").toVectorReal();
  int nbeats = tempoHints.size();
  if (nbeats > 2) {
    // get the average period in these beats
    Real periodHint = (tempoHints[nbeats-1] - tempoHints[0]) / (nbeats-1);

    // convert bpm to lag
    periodHint = periodHint / _frameTime;

    // check for validity
    if ((_minLag < periodHint) && (periodHint < _maxLag)) {
      // use a gaussian model
      Real spread = 20; // WARNING: MAGIC NUMBER
      for (int i=0; i<int(_weighting.size()); ++i) {
        Real dev = i - periodHint;
        _weighting[i] = exp(-dev*dev / (spread*spread));
      }
    }
    else {
      throw EssentiaException("TempoTap: tempoHints were not found to match any realistic BPM value");
    }
  }

  // configure the peak detection algorithm to find the index of the highest peak
  _peakDetector->configure("range", _comblen - 1,
                           "minPosition", 0,
                           "maxPosition", _comblen,
                           "orderBy", "amplitude",
                           "maxPeaks", 1,
                           "interpolate", true);

  _peakDetector->output("positions").set(_peaksPositions);
  _peakDetector->output("amplitudes").set(_peaksMagnitudes);

  reset();
}




void TempoTap::compute() {
  const std::vector<Real>& featuresFrame = _featuresFrame.get();
  vector<Real>& period = _periods.get();
  vector<Real>& phases = _phases.get();

  // buffer new frame of features
  _featuresNew.push_back(featuresFrame);

  const int nframe = _featuresNew.size();

  // we first need to buffer enough frames to be able to compute the autocorrelation
  // If we have not accumulated enough frames, we just return an empty period and phase
  // vector, otherwise we do the computation and reset our internal memory at the end.
  if (nframe < _frameHop) {
    _acf.clear();
    period.clear();
    phases.clear();
    return;
  }


  // fill in the buffer of features on which to perform the analysis
  // featuresBuffer = _featuresOld[0:_numberFrames - _frameHop] + _featuresNew[0:_frameHop];
  vector<vector<Real> > featuresBuffer(_featuresOld.size() + _featuresNew.size());
  int j = 0;
  for (int i=0; i<int(_featuresOld.size()); ++i, ++j) {
    featuresBuffer[j] = _featuresOld[i];
  }
  for (int i=0; i<int(_featuresNew.size()); ++i, ++j) {
    featuresBuffer[j] = _featuresNew[i];
  }

  // update the memory buffer
  // _featuresOld = _featuresOld[frameHop:end] + featuresNew;
  int endBuf = max((int)_featuresOld.size() - _frameHop, 0);
  for (int i=0; i<endBuf; i++) {
    _featuresOld[i] = _featuresOld[i+_frameHop];
  }
  for (int i=endBuf; i<(int)_featuresOld.size(); i++) {
    _featuresOld[i] = _featuresNew[i - (int)_featuresOld.size() + _frameHop];
  }

  vector<vector<Real> > features = transpose(featuresBuffer);

  computePeriods(features);
  computePhases(features);

  // empty features after computation
  _featuresNew.clear();
}

void TempoTap::computePeriods(const vector<vector<Real> >& features) {
  vector<Real>& period = _periods.get();
  int nfeats = features.size();

  // compute the autocorrelation of each feature
  _acf.resize(nfeats);
  for (int i=0; i<nfeats; i++) {
    _autocorr->input("array").set(features[i]);
    _autocorr->output("autoCorrelation").set(_acf[i]);
    _autocorr->compute();
  }

  // compute the multi comb filter over 4 different peaks in the acfs
  // weighting the contribution of different periods with a rayleigh
  // distribution
  period.resize(nfeats);
  _mcomb.resize(nfeats);

  for (int f=0; f<nfeats; f++) {
    vector<Real>& mcomb = _mcomb[f];
    mcomb = vector<Real>(_comblen, 0.0);

    for (int i=1; i<_comblen-1; i++) {
      for (int a=1; a<_maxelem+1; a++) {
        for (int b=1-a; b<a; b++) {
          assert(a * (i+1) + b-1 >= 0);
          assert(a * (i+1) + b-1 < (int)_acf[f].size());
          mcomb[i] += _acf[f][a * (i+1) + b-1 ] * _weighting[i] / (2*a - 1);
        }
      }
    }

    // get max peak
    _peakDetector->input("array").set(mcomb);
    _peakDetector->compute();

    if (!_peaksPositions.empty()) {
      period[f] = _peaksPositions[0];
    }
    else {
      // should we throw an exception here?
      period[f] = 0.;
    }
  }
}


void TempoTap::computePhases(const vector<vector<Real> >& features) {
  vector<Real>& phases = _phases.get();
  vector<Real>& period = _periods.get();

  int nfeats = features.size();
  int nframe = features.empty() ? 0 : features[0].size();

  // filter the features vector through a comb filter at the estimated
  // period, looking for the initial phase of beat locations in this buffer
  int philen = (int)round(nframe / (Real)_nPeaks);

  _phasesOut.resize(nfeats);
  phases.resize(nfeats);
  for (int f=0; f<nfeats; f++) {
    vector<Real>& phasesOut = _phasesOut[f];

    // only try to find the phase if the period is within the specified acceptable range
    if (_minLag < period[f] && period[f] < _maxLag) {
      phasesOut = vector<Real>(philen, 0.0);

      for (int i=0; i<philen; i++) {
        for (int a=0; a<_nPeaks; a++)  {
          int idx = (int)round(a*period[f] + i);
          assert(idx >= 0);
          assert(idx < (int)features[f].size());
          phasesOut[i] += features[f][idx];
        }
      }

      _peakDetector->input("array").set(phasesOut);
      _peakDetector->compute();

      if (!_peaksPositions.empty()) {
        phases[f] = _peaksPositions[0];
      }
      else {
        phases[f] = -1;
      }

      //while (phases[f] >= period[f])
      while (phases[f] >= period[f] && period[f] > _minLag) { // strange (useless) test... we already know that period[f] > minLag
        phases[f] -= period[f];
      }
    } else {
      // the period was out of range, do not look for phases
      // should we throw an exception here? probably not because it could happen
      // that a single frame produces no estimates, and this shouldn't stop the
      // whole processing...
      period[f] = 0;
      phases[f] = -1;
    }
  }
}
