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

#include "tensorflowinputvggish.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowInputVGGish::name = "TensorflowInputVGGish";
const char* TensorflowInputVGGish::category = "Spectral";
const char* TensorflowInputVGGish::description = DOC(
  "This algorithm computes mel-bands with a particular parametrization specific to VGGish based models [1, 2, 3].\n"
  "\n"
  "References:\n"
  "  [1] Gemmeke, J. et. al., AudioSet: An ontology and human-labelled dataset for audio events, ICASSP 2017\n"
  "  [2] Hershey, S. et. al., CNN Architectures for Large-Scale Audio Classification, ICASSP 2017\n"
  "  [3] Supported models at https://essentia.upf.edu/models/");


void TensorflowInputVGGish::configure() {

  // Analysis parameters are hardcoded to make sure they match the values used on training:
  // https://github.com/tensorflow/models/tree/master/research/audioset/vggish
  int frameSize = 400;
  int fftSize = 512;
  int numberBands=64;
  Real sampleRate = 16000.0;
  Real lowFrequencyBound = 125.0;
  Real highFrequencyBound = 7500.0;
  string warpingFormula = "htkMel";
  string type = "magnitude";
  string weighting = "warping";
  string normalize = "unit_max";
  Real shift = 0.01;
  string comp = "log";

  _windowing->configure("normalized", false,
                        "zeroPadding", fftSize-frameSize,
                        "zeroPhase", false);

  _spectrum->configure("size", fftSize);

  _melBands->configure("inputSize", fftSize / 2 + 1,
                       "numberBands", numberBands,
                       "sampleRate", sampleRate,
                       "lowFrequencyBound", lowFrequencyBound,
                       "highFrequencyBound", highFrequencyBound,
                       "warpingFormula", warpingFormula,
                       "weighting", weighting,
                       "type", type,
                       "normalize", normalize);

  _shift->configure("shift", shift);

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


void TensorflowInputVGGish::compute() {
  const std::vector<Real>& frame = _frame.get();

  if (frame.size() != 400) {
    throw(EssentiaException("TensorflowInputVGGish: This algorithm only accepts input frames of size 400."));
  }

  _windowing->input("frame").set(frame);
  _compression->output("array").set(_bands.get());

  _windowing->compute();
  _spectrum->compute();
  _melBands->compute();

  // HTK excludes the spectrogram DC bin; make sure it always gets a zero
  // coefficient.
  // _melBandsFrame[0] = 0.f;

  _shift->compute();
  _compression->compute();
}
