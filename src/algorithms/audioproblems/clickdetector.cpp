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

#include "clickdetector.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char *ClickDetector::name = "ClickDetector";
const char *ClickDetector::category = "Audio Problems";
const char *ClickDetector::description = DOC("This algorithm detects the locations of impulsive noises (clicks and pops) on the input audio frame."
" It relies on LPC coefficients to inverse-filter the audio in order to attenuate the stationary part and enhance the prediction error"
" (or excitation noise)[1]. After this, a matched filter is used to further enhance the impulsive peaks." 
" The detection threshold is obtained from a robust estimate of the excitation noise power [2] plus a parametric gain value.\n"
"\n"
"References:\n"
"[1] Vaseghi, S. V., & Rayner, P. J. W. (1990). Detection and suppression of impulsive noise in speech communication systems."
" IEE Proceedings I (Communications, Speech and Vision), 137(1), 38-46."
"\n"
"[2] Vaseghi, S. V. (2008). Advanced digital signal processing and noise reduction. John Wiley & Sons. Page 355");


void ClickDetector::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _order = parameter("order").toInt();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _detectionThld = db2pow(parameter("detectionThreshold").toReal());
  _powerEstimationThld = parameter("powerEstimationThreshold").toReal();
  _silenceThld = db2pow(parameter("silenceThreshold").toReal());

  _LPC->configure(INHERIT("order"));

  if (_frameSize <= _order)
    throw(
      EssentiaException("ClickDetector: the number of LPC coefficientes has to be smaller "
                        "than the size of the input frame"));

  if (_frameSize < _hopSize)
    throw(EssentiaException(
      "ClickDetector: hopSize has to be smaller or equal than the input frame size"));

  _startProc = int(_frameSize / 2 - _hopSize / 2);
  _endProc = int(_frameSize / 2 + _hopSize / 2);

  if (_startProc < (uint)_order) {
    uint unproc = _order - _startProc;
    uint maxHop = _frameSize - 2 * _order;
    E_INFO("ClickDetector: non-optimal 'HopSize' parameter. The " << unproc << " first samples will not be processed."
    " To prevent this problem use a maximum 'HopSize' of " << maxHop);
    _startProc = _order;
  }

  _idx = 0;
}


void ClickDetector::compute() {
  const std::vector<Real> frame = _frame.get();
  std::vector<Real> &clickStarts = _clickStarts.get();
  std::vector<Real> &clickEnds = _clickEnds.get();


  if (instantPower(frame) <_silenceThld) {
    _idx += 1;
    return;
  }

  std::vector<Real> lpcCoeff(_order, 0.f);
  std::vector<Real> matchedCoeff(_order, 0.f);
  std::vector<Real> reflectionCoeff;
  _LPC->input("frame").set(frame);
  _LPC->output("lpc").set(lpcCoeff);
  _LPC->output("reflection").set(reflectionCoeff);
  _LPC->compute();

  // It was found that with the raw coefficients the output of the matched filter could be amplified up to 40dB.
  // Normalization of the coefficients keeps the filtered signals on the same range without a perceived difference
  // in the peak enhancement. 
  normalize(lpcCoeff);

  _InverseFilter->configure("numerator", lpcCoeff);

  // It is not necessary to process the overlapping part of the signal.
  std::vector<Real> subframe(frame.begin() + _startProc - _order, frame.begin() + _endProc + _order);
  std::vector<Real> e;
  _InverseFilter->input("signal").set(subframe);
  _InverseFilter->output("signal").set(e);
  _InverseFilter->compute();

  std::vector<Real> eInv = e;
  std::reverse(eInv.begin(), eInv.end());

  for (uint i = 0; i < matchedCoeff.size(); i++)
    matchedCoeff[i] = -lpcCoeff[i];

  std::vector<Real> eMF;
  _MatchedFilter->configure("numerator", matchedCoeff);
  _MatchedFilter->input("signal").set(eInv);
  _MatchedFilter->output("signal").set(eMF);
  _MatchedFilter->compute();

  std::reverse(eMF.begin(), eMF.end());

  Real robustPowerValue = robustPower(e, _powerEstimationThld) * _detectionThld;

  Real threshold = std::max(robustPowerValue, _silenceThld);

  std::vector<uint> detections;
  for (uint i = _order; i < eMF.size() - _order; i++)
    if (pow(eMF[i], 2.0) >= threshold)
      detections.push_back(_startProc + i - _order);

  if (detections.size() >= 1) {
    clickStarts.push_back((detections[0] + _idx * _hopSize) /_sampleRate);
    uint end = detections[0];

    for (uint i = 1; i < detections.size(); i++){
      if (detections[i] == (detections[i - 1] + 1)) 
        end = detections[i];
      else {
        clickEnds.push_back((end + _idx * _hopSize) / _sampleRate);
        clickStarts.push_back((detections[i] + _idx * _hopSize) / _sampleRate);
        end = detections[i];
      }
    }
    clickEnds.push_back((end + _idx * _hopSize) / _sampleRate);
  } 

  _idx += 1;
}


void ClickDetector::reset() {
  _idx = 0;
}


Real ClickDetector::robustPower(std::vector<Real> x, Real k ) {
  for (uint i = 0; i < x.size(); i ++)
    x[i] *= x[i];

  Real medianValue = median(x);

  _Clipper->configure("max", medianValue * k);
  
  std::vector<Real> robustPowerX;
  _Clipper->input("signal").set(x);
  _Clipper->output("signal").set(robustPowerX);
  _Clipper->compute();

  Real robustPower = mean(robustPowerX);

  return robustPower;
}
