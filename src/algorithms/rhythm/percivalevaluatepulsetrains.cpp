/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "percivalevaluatepulsetrains.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;

namespace essentia {
namespace standard {

const char* PercivalEvaluatePulseTrains::name = "PercivalEvaluatePulseTrains";
const char* PercivalEvaluatePulseTrains::category = "Rhythm";
const char* PercivalEvaluatePulseTrains::description = DOC("This algorithm implements the 'Evaluate Pulse Trains' step as described in [1]."
"Given an input onset detection function (ODF, called \"onset strength signal\", OSS, in the original paper) and a number of candidate BPM peak positions, the ODF is correlated with ideal expected pulse "
"trains (for each candidate tempo lag) shifted in time by different amounts."
"The candidate tempo lag that generates a periodic pulse train with the best correlation to the ODF is returned as the best tempo estimate.\n"
"For more details check the referenced paper."
"Please note that in the original paper, the term OSS (Onset Strength Signal) is used instead of ODF."
"\n"
"\n"
"References:\n"
"  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.\n"
"  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.\n\n");

void PercivalEvaluatePulseTrains::configure() {
}

void PercivalEvaluatePulseTrains::calculatePulseTrains(const std::vector<Real>& ossWindow,
                                               const int lag,
                                               Real& magScore,
                                               Real& varScore) {
  vector<Real> bpMagnitudes;
  int period = lag;
  bpMagnitudes.resize(lag);
  for (int phase=0; phase < period; ++phase) {
    Real currentMagScore;
    currentMagScore = 0.0;
    for (int b=0; b < 4; ++b) {
      int ind;
      ind = (int)(phase + b * period);
      if (ind >= 0) {
        currentMagScore += ossWindow[ind];
      }
      ind = (int)(phase + b * period * 2);
      if (ind >= 0) {
        currentMagScore += 0.5 * ossWindow[ind];
      }
      ind = (int)(phase + b * period * 3 / 2);
      if (ind >= 0) {
        currentMagScore += 0.5 * ossWindow[ind];
      }
    }
    bpMagnitudes[phase] = currentMagScore;
  }
  magScore = *std::max_element(bpMagnitudes.begin(), bpMagnitudes.end());
  varScore = variance(bpMagnitudes, mean(bpMagnitudes));
}

void PercivalEvaluatePulseTrains::compute() {
  const vector<Real>& oss = _oss.get();
  const vector<Real>& peakPositions = _peakPositions.get();
  Real& lag = _lag.get();

  if (peakPositions.size() == 0) {
    // No peaks have been detected, return lag -1
    lag = -1;
    return;
  }

  vector<Real> tempoScores;
  vector<Real> onsetScores;
  tempoScores.reserve(peakPositions.size());
  onsetScores.reserve(peakPositions.size());
  for (int i=0; i<(int)peakPositions.size(); ++i) {
    Real candidate = peakPositions[i];
    if (candidate != 0) {
      int lag = (int) round(candidate);
      Real magScore = 0;
      Real varScore = 0;
      if (lag > 0) {
        // Passing lag = 0 to calculatePulseTrains will result in crash
        calculatePulseTrains(oss, lag, magScore, varScore);
      }
      tempoScores[i] = magScore;
      onsetScores[i] = varScore;
    }
  }
  vector<Real> comboScores;
  comboScores.resize(peakPositions.size());
  Real sumTempoScores = sum(tempoScores);
  Real sumOnsetScroes = sum(onsetScores);
  for (int i=0; i<(int)peakPositions.size(); ++i) {
    comboScores[i] = tempoScores[i]/sumTempoScores + onsetScores[i]/sumOnsetScroes;
  }
  // NOTE: original python implementation normalizes comboScores (like tempoScores and onsetScore).
  // As we are only taking argmax, we assume there is no need for this normalization.
  Real bestScorePosition = argmax(comboScores);
  lag = round(peakPositions[bestScorePosition]);
}

} // namespace standard
} // namespace essentia
