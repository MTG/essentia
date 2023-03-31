/*
 * Copyright (C) 2006-2023  Music Technology Group - Universitat Pompeu Fabra
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

#include "tensorflowinputfsdsinet.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowInputFSDSINet::name = "TensorflowInputFSDSINet";
const char* TensorflowInputFSDSINet::category = "Spectral";
const char* TensorflowInputFSDSINet::description = DOC(
  "This algorithm computes mel bands from an audio frame with the specific parametrization required by the FSD-SINet models."
  "\n\n"
  "References:\n"
  "  [1] Fonseca, E., Ferraro, A., & Serra, X. (2021). Improving sound event classification by increasing shift invariance in convolutional neural networks. arXiv preprint arXiv:2107.00623.\n"
  "  [2] https://github.com/edufonseca/shift_sec"
);


void TensorflowInputFSDSINet::configure() {

  _windowing->configure(
    "type", "hamming",
    "zeroPadding", _zeroPadding, 
    "normalized", false,
    "constantsDecimals", 2,
    "zeroPhase", false,
    "splitPadding", true,
    "symmetric", false
  );

  _spectrum->configure("size", _fftSize);

  _melBands->configure(
    "inputSize", _spectrumSize,
    "numberBands", 96,
    "sampleRate", 22050,
    "lowFrequencyBound", 50,
    "highFrequencyBound", 10500,
    "warpingFormula", "slaneyMel",
    "weighting", "linear",
    "normalize", "unit_max",
    "type", "power",
    "log", false
  );

  _compression->configure("type", "log10");


  // Set the intermediate buffers.
  _windowing->output("frame").set(_windowedFrame);

  _spectrum->input("frame").set(_windowedFrame);
  _spectrum->output("spectrum").set(_spectrumFrame);

  _melBands->input("spectrum").set(_spectrumFrame);
  _melBands->output("bands").set(_melBandsFrame);

  _compression->input("array").set(_melBandsFrame);
}


void TensorflowInputFSDSINet::compute() {
  const std::vector<Real>& frame = _frame.get();

  if ((int)frame.size() != _frameSize) {
    throw(EssentiaException("TensorflowInputFSDSINet: This algorithm only accepts input frames of size 660."));
  }

  _windowing->input("frame").set(frame);
  _compression->output("array").set(_bands.get());

  _windowing->compute();
  _spectrum->compute();
  _melBands->compute();
  _compression->compute();
}
