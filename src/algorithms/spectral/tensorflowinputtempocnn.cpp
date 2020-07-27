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

#include "tensorflowinputtempocnn.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowInputTempoCNN::name = "TensorflowInputTempoCNN";
const char* TensorflowInputTempoCNN::category = "Spectral";
const char* TensorflowInputTempoCNN::description = DOC(
  "This algorithm computes mel-bands specific to the input of TempoCNN-based models.\n"
  "\n"
  "References:\n"
  "  [1] Hendrik Schreiber, Meinard Müller, A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), Paris, France, Sept. 2018.\n"
  "  [2] Hendrik Schreiber, Meinard Müller, Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters Proceedings of the Sound and Music Computing Conference (SMC), Málaga, Spain, 2019.\n"
  "  [3] Original models and code at https://github.com/hendriks73/tempo-cnn\n"
  "  [4] Supported models at https://essentia.upf.edu/models/");


void TensorflowInputTempoCNN::configure() {

  // Analysis parameters are hardcoded to make sure they match the values used on training:
  // https://github.com/hendriks73/tempo-cnn/blob/master/tempocnn/feature.py
  int frameSize = 1024;
  int numberBands=40;
  Real sampleRate = 11025.0;
  Real lowFrequencyBound = 20;
  Real highFrequencyBound = 5000;
  string warpingFormula = "slaneyMel";
  string weighting = "linear";
  string normalize = "unit_tri";
  string type = "magnitude";

  _windowing->configure("normalized", false);

  _spectrum->configure("size", frameSize);

  _melBands->configure("inputSize", frameSize / 2 + 1,
                       "numberBands", numberBands,
                       "sampleRate", sampleRate,
                       "lowFrequencyBound", lowFrequencyBound,
                       "highFrequencyBound", highFrequencyBound,
                       "warpingFormula", warpingFormula,
                       "weighting", weighting,
                       "normalize", normalize,
                       "type", type);

  // Set the intermediate buffers.
  _windowing->output("frame").set(_windowedFrame);

  _spectrum->input("frame").set(_windowedFrame);
  _spectrum->output("spectrum").set(_spectrumFrame);

  _melBands->input("spectrum").set(_spectrumFrame);
}


void TensorflowInputTempoCNN::compute() {
  const std::vector<Real>& frame = _frame.get();

  if (frame.size() != 1024) {
    throw(EssentiaException("TensorflowInputTempoCNN: This algorithm only accepts input frames of size 1024."));
  }

  _windowing->input("frame").set(frame);
  _melBands->output("bands").set(_bands.get());

  _windowing->compute();
  _spectrum->compute();
  _melBands->compute();
}
