/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#include "tensorflowinputmusicnn.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowInputMusiCNN::name = "TensorflowInputMusiCNN";
const char* TensorflowInputMusiCNN::category = "Spectral";
const char* TensorflowInputMusiCNN::description = DOC(
  "This algorithm computes mel-bands with a particular parametrization specific to MusiCNN based models [1, 2].\n"
  "\n"
  "References:\n"
  "  [1] Pons, J., & Serra, X. (2019). musicnn: Pre-trained convolutional neural networks for music audio tagging. arXiv preprint arXiv:1909.06654.\n"
  "  [2] Supported models at https://essentia.upf.edu/models/");


void TensorflowInputMusiCNN::configure() {

  // Analysis parameters are hardcoded to make sure they match the values used on training:
  // https://github.com/jordipons/musicnn-training/blob/master/src/config_file.py
  int frameSize = 512;
  int numberBands=96;
  Real sampleRate = 16000.0;
  string warpingFormula = "slaneyMel";
  string weighting = "linear";
  string normalize = "unit_tri";
  Real shift = 1;
  Real scale = 10000;
  string comp = "log10";

  _windowing->configure("normalized", false);

  _spectrum->configure("size", frameSize);

  _melBands->configure("inputSize", frameSize / 2 + 1,
                       "numberBands", numberBands,
                       "sampleRate", sampleRate,
                       "highFrequencyBound", sampleRate / 2,
                       "warpingFormula", warpingFormula,
                       "weighting", weighting,
                       "normalize", normalize);

  _shift->configure("shift", shift, "scale", scale);

  _compression->configure("type", comp);

  // Set the intermediate buffers.
  _windowing->output("frame").set(_windowedFrame);

  _spectrum->input("frame").set(_windowedFrame);
  _spectrum->output("spectrum").set(_spectrumFrame);

  _melBands->input("spectrum").set(_spectrumFrame);
  _melBands->output("bands").set(_melBandsFrame);

  _shift->input("array").set(_melBandsFrame);
  _shift->output("array").set(_shiftedFrame);

  _compression->input("array").set(_shiftedFrame);
}


void TensorflowInputMusiCNN::compute() {
  const std::vector<Real>& frame = _frame.get();

  if (frame.size() != 512) {
    throw(EssentiaException("TensorflowInputMusiCNN: This algorithm only accepts input frames of size 512."));
  }

  _windowing->input("frame").set(frame);
  _compression->output("array").set(_bands.get());

  _windowing->compute();
  _spectrum->compute();
  _melBands->compute();
  _shift->compute();
  _compression->compute();
}
