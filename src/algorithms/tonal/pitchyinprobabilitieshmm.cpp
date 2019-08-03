/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

#include "pitchyinprobabilitieshmm.h"
#include "essentiamath.h"
#include <time.h>

using namespace std;
using namespace essentia;
using namespace standard;


const char* PitchYinProbabilitiesHMM::name = "PitchYinProbabilitiesHMM";
const char* PitchYinProbabilitiesHMM::category = "Pitch";
const char* PitchYinProbabilitiesHMM::description = DOC("This algorithm estimates the smoothed fundamental frequency given the pitch candidates and probabilities using hidden Markov models. It is a part of the implementation of the probabilistic Yin algorithm [1].\n"
"\n"
"An exception is thrown if an empty signal is provided.\n"
"\n"
"References:\n"
"  [1] M. Mauch and S. Dixon, \"pYIN: A Fundamental Frequency Estimator\n"
"  Using Probabilistic Threshold Distributions,\" in Proceedings of the\n"
"  IEEE International Conference on Acoustics, Speech, and Signal Processing\n"
"  (ICASSP 2014)Project Report, 2004");

void PitchYinProbabilitiesHMM::configure() {
  _viterbi->configure();

  _minFrequency = parameter("minFrequency").toReal();
  _numberBinsPerSemitone = parameter("numberBinsPerSemitone").toInt();
  _selfTransition = parameter("selfTransition").toReal();
  _yinTrust = parameter("yinTrust").toReal();

  _transitionWidth = 5 * (_numberBinsPerSemitone / 2) + 1;
  _nPitch = 69 * _numberBinsPerSemitone;
  _freqs = vector<Real>( 2 * _nPitch);
  for (size_t iPitch = 0; iPitch < _nPitch; ++iPitch) {
      _freqs[iPitch] = _minFrequency * pow(2, iPitch * 1.0 / (12 * _numberBinsPerSemitone));
      _freqs[iPitch + _nPitch] = -_freqs[iPitch];
  }

  _init.clear();
  _from.clear();
  _to.clear();
  _transProb.clear();

  // INITIAL VECTOR
  _init = vector<Real>(2 * _nPitch, 1.0 / 2 * _nPitch);
  
  // TRANSITIONS
  for (size_t iPitch = 0; iPitch < _nPitch; ++iPitch)
  {
    int theoreticalMinNextPitch = static_cast<int>(iPitch)-static_cast<int>(_transitionWidth / 2);
    size_t minNextPitch = iPitch > _transitionWidth/2 ? iPitch - _transitionWidth / 2 : 0;
    size_t maxNextPitch = iPitch < _nPitch - _transitionWidth / 2 ? iPitch + _transitionWidth / 2 : _nPitch - 1;
    
    // WEIGHT VECTOR
    Real weightSum = 0;
    vector<Real> weights;
    for (size_t i = minNextPitch; i <= maxNextPitch; ++i)
    {
      if (i <= iPitch)
      {
          weights.push_back(i - theoreticalMinNextPitch + 1);
      } else {
          weights.push_back(iPitch - theoreticalMinNextPitch + 1 - (i - iPitch));
      }
      weightSum += weights[weights.size()-1];
    }
    
    // TRANSITIONS TO CLOSE PITCH
    for (size_t i = minNextPitch; i <= maxNextPitch; ++i)
    {
      _from.push_back(iPitch);
      _to.push_back(i);
      _transProb.push_back(weights[i - minNextPitch] / weightSum * _selfTransition);

      _from.push_back(iPitch);
      _to.push_back(i + _nPitch);
      _transProb.push_back(weights[i - minNextPitch] / weightSum * (1 - _selfTransition));

      _from.push_back(iPitch + _nPitch);
      _to.push_back(i + _nPitch);
      _transProb.push_back(weights[i - minNextPitch] / weightSum * _selfTransition);
      
      _from.push_back(iPitch + _nPitch);
      _to.push_back(i);
      _transProb.push_back(weights[i - minNextPitch] / weightSum * (1 - _selfTransition));
    }
  }
}

const vector<Real> PitchYinProbabilitiesHMM::calculateObsProb(const vector<Real> pitchCandidates, const vector<Real> probabilities) {
  
  vector<Real> out = vector<Real>(2 * _nPitch + 1);
  Real probYinPitched = 0;
  // BIN THE PITCHES
  for (size_t iPair = 0; iPair < pitchCandidates.size(); ++iPair) {
    Real freq = 440. * pow(2, (pitchCandidates[iPair] - 69)/12);
    if (freq <= _minFrequency) continue;
    Real d = 0;
    Real oldd = 1000;
    for (size_t iPitch = 0; iPitch < _nPitch; ++iPitch) {
      d = abs(freq - _freqs[iPitch]);
      if (oldd < d && iPitch > 0) {
        // previous bin must have been the closest
        out[iPitch-1] = probabilities[iPair];
        probYinPitched += out[iPitch-1];
        break;
      }
      oldd = d;
    }
  }

  Real probReallyPitched = _yinTrust * probYinPitched;

  for (size_t iPitch = 0; iPitch < _nPitch; ++iPitch) {
      if (probYinPitched > 0) out[iPitch] *= (probReallyPitched/probYinPitched);
      out[iPitch + _nPitch] = (1 - probReallyPitched) / _nPitch;
  }

  return(out);
}

void PitchYinProbabilitiesHMM::compute() {
  const vector<vector<Real> >& pitchCandidates = _pitchCandidates.get();
  const vector<vector<Real> >& probabilities = _probabilities.get();

  if (pitchCandidates.empty() || probabilities.empty()) {
    throw EssentiaException("PitchYin: Cannot compute pitch detection on empty inputs.");
  }

  vector<Real>& pitch = _pitch.get();
  
  vector<vector<Real> > obsProb(pitchCandidates.size());
  for (size_t iFrame = 0; iFrame < pitchCandidates.size(); ++iFrame) {
      obsProb[iFrame] = calculateObsProb(pitchCandidates[iFrame], probabilities[iFrame]);
  }

  vector<int> path;
  _viterbi->input("observationProbabilities").set(obsProb);
  _viterbi->input("initialization").set(_init);
  _viterbi->input("fromIndex").set(_from);
  _viterbi->input("toIndex").set(_to);
  _viterbi->input("transitionProbabilities").set(_transProb);
  _viterbi->output("path").set(path);
  _viterbi->compute();

  _tempPitch.resize(path.size());

  // time(&start1);
  for (size_t iFrame = 0; iFrame < path.size(); ++iFrame)
  {
    Real hmmFreq = _freqs[path[iFrame]];
    Real bestFreq = 0;
    Real leastDist = 10000;
    if (hmmFreq > 0)
    {
      for (size_t iPitch = 0; iPitch < pitchCandidates[iFrame].size(); ++iPitch)
      {
        Real freq = 440. * pow(2, (pitchCandidates[iFrame][iPitch] - 69) / 12);
        Real dist = abs(hmmFreq - freq);
        if (dist < leastDist) {
          leastDist = dist;
          bestFreq = freq;
        }
      }
    } else {
      bestFreq = hmmFreq;
    }
    _tempPitch[iFrame] = bestFreq;
  }
  pitch = _tempPitch;
}
