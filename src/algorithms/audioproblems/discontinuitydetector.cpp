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

#include "discontinuitydetector.h"

using namespace essentia;
using namespace standard;

const char *DiscontinuityDetector::name = "DiscontinuityDetector";
const char *DiscontinuityDetector::category = "Audio Problems";
const char *DiscontinuityDetector::description =
    DOC("This algorithm uses LPC and some heuristics to detect discontinuities "
        "in an audio signal. [1].\n"
        "\n"
        "References:\n"
        "  [1] Mühlbauer, R. (2010). Automatic Audio Defect Detection.\n");


void DiscontinuityDetector::configure() {
  _order = parameter("order").toInt();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _kernelSize = parameter("kernelSize").toInt();
  _detectionThld = parameter("detectionThreshold").toReal();
  _energyThld = parameter("energyThreshold").toReal();
  _subFrameSize = parameter("subFrameSize").toInt();
  _silenceThld = db2pow(parameter("silenceThreshold").toReal());

  _medianFilter->configure(INHERIT("kernelSize"));
  _LPC->configure(INHERIT("order"));
  _windowing->configure("size", _frameSize, "zeroPhase", false, "type",
                        "triangular");

  if (_frameSize <= _order)
    throw(
        EssentiaException("DiscontinuityDetector: the number of LPC coefficientes has to be smaller "
                          "than the size of the input frame"));

  if (_frameSize < _hopSize)
    throw(EssentiaException(
        "DiscontinuityDetector: hopSize has to be smaller or equal than the input frame size"));

  if (_frameSize < _kernelSize)
    throw(EssentiaException(
        "DiscontinuityDetector: kernelSize has to be smaller or equal than the input frame size"));

  if (_frameSize < _subFrameSize)
    throw(EssentiaException(
        "DiscontinuityDetector: subFrameSize has to be smaller than the input frame size"));
}


void DiscontinuityDetector::compute() {
  const std::vector<Real> frame = _frame.get();
  std::vector<Real> &discontinuityLocations = _discontinuityLocations.get();
  std::vector<Real> &discontinuityAmplitues = _discontinuityAmplitues.get();
  std::vector<Real> frameAux = frame;

  if (instantPower(frameAux) < _silenceThld) return;

  int inputSize = frameAux.size();

  if (inputSize <= _order)
    throw(
        EssentiaException("DiscontinuityDetector: the number of LPC coefficientes has to be smaller "
                          "than the size of the input frame"));

  if (inputSize < _hopSize)
    throw(EssentiaException("DiscontinuityDetector: hopSize has to be smaller than the input frame size"));

  if (inputSize < _kernelSize)
    throw(
        EssentiaException("DiscontinuityDetector: kernelSize has to be smaller than the input frame size"));

  if (inputSize < _subFrameSize)
    throw(EssentiaException(
        "DiscontinuityDetector: subFrameSize has to be smaller than the input frame size"));

  if (inputSize != _frameSize) {
    _frameSize = inputSize;
    _windowing->configure("size", _frameSize, "zeroPhase", false, "type",
                          "triangular");
  }

  // the analysisSize is set so the the same region is never processed twice on
  // contiguous frames
  int start = inputSize / 2 - _hopSize / 2;
  int end = inputSize / 2 + _hopSize / 2;

  if (inputSize == _hopSize) start = _order;

  int analysisSize = end - start;

  std::vector<Real> frameProc(_frameSize);
  _windowing->input("frame").set(frameAux);
  _windowing->output("frame").set(frameProc);
  _windowing->compute();

  normalizeAbs(frameProc);

  std::vector<Real> lpc_coeff;
  std::vector<Real> reflection_coeff;

  _LPC->input("frame").set(frameProc);
  _LPC->output("lpc").set(lpc_coeff);
  _LPC->output("reflection").set(reflection_coeff);
  _LPC->compute();

  lpc_coeff.erase(lpc_coeff.begin(), lpc_coeff.begin() + 1);

  std::vector<Real> error(analysisSize, 0.f);
  std::vector<Real> predAux(_order, 0.f);
  Real prediction;
  int idx = 0;

  for (int i = start; i < end; i++, idx++) {
    for (int j = 0; j < _order; j++) {
      predAux[j] = lpc_coeff[j] * frameProc[i - j - 1];
    }
    prediction = -sum(predAux);
    error[idx] = abs(prediction - frameProc[i]);
  }

  // A median filter cleans up the error signal to focus on the narrow peaks.
  std::vector<Real> medianFilter;

  _medianFilter->input("array").set(error);
  _medianFilter->output("filteredArray").set(medianFilter);
  _medianFilter->compute();

  std::vector<Real> filteredError(analysisSize, 0.f);

  for (int i = 0; i < analysisSize; i++)
    filteredError[i] = abs(error[i] - medianFilter[i]);

  // Use only the non-silent subframes for the threshold computation.
  // Otherwise they can lower it too much.
  std::vector<Real> subFrame(_subFrameSize, 0.f);
  std::vector<Real> masked;
  std::vector<Real>::const_iterator inputIt = frame.begin() + start;

  for (int i = 0; i <= analysisSize - _subFrameSize; i += _subFrameSize) {
    subFrame.assign(inputIt + i, inputIt + i + _subFrameSize);
    if (instantPower(subFrame) > db2amp(_energyThld))
      masked.insert(masked.end(), error.begin() + i,
                    error.begin() + i + _subFrameSize);
  }

  if (masked.size() == 0) return;

  // The threshold goes up if rather the std or the median of the error signal
  // are high.
  float threshold =
      _detectionThld * (stddev(masked, mean(masked)) + median(masked));

  for (int i = 0; i < analysisSize; i++) {
    if (filteredError[i] >= threshold) {
      discontinuityLocations.push_back((Real)(i + start));
      discontinuityAmplitues.push_back(filteredError[i]);
    }
  }
}
