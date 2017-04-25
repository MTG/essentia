/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "nsgconstantqstreaming.h"
#include "essentia.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* NSGConstantQStreaming::name = "NSGConstantQStreaming";
const char* NSGConstantQStreaming::category = "Streaming";
const char* NSGConstantQStreaming::description = DOC("This algorithm computes a constant Q transform using non stationary Gabor frames and returns a complex time-frequency representation of the input vector.\n"
"The implementation is inspired by the toolbox described in [1]."
"\n"
"References:\n"
    "[1] Schörkhuber, C., Klapuri, A., Holighaus, N., \& Dörfler, M. (n.d.). A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.");

NSGConstantQStreaming::NSGConstantQStreaming() : AlgorithmComposite() {
  declareInput(_signal, "frame", "the input frame (vector)");
  declareOutput(_constantQ, "constantq", "the constant Q transform of the input frame");
  declareOutput(_constantQDC, "constantqdc", "the DC band transform of the input frame. Only needed for the inverse transform");
  declareOutput(_constantQNF, "constantqnf", "the Nyquist band transform of the input frame. Only needed for the inverse transform");
  _wrapper =  AlgorithmFactory::create("NSGConstantQ");

  _signal                         >> _wrapper->input("frame");
  //_wrapper->output("constantq")   >> _constantQ;
  _wrapper->output("constantqdc") >> _constantQDC;
  _wrapper->output("constantqnf") >> _constantQNF;

}

void NSGConstantQStreaming::configure() {
  //_wrapper->configure();

  _wrapper->configure(INHERIT("sampleRate"),
                      INHERIT("minFrequency"),
                      INHERIT("maxFrequency"),
                      INHERIT("binsPerOctave"),
                      INHERIT("gamma"),
                      INHERIT("inputSize"),
                      INHERIT("rasterize"),
                      INHERIT("phaseMode"),
                      INHERIT("normalize"),
                      INHERIT("minimumWindow"),
                      INHERIT("windowSizeFactor"));

}

AlgorithmStatus NSGConstantQStreaming::process() {


  if (!shouldStop()) return PASS;
  _wrapper->process();
  //_wrapper->input("frame").acquire();
  //_wrapper->process();
  return FINISHED;
}


}
}
