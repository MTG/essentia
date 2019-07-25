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
const char* NSGConstantQStreaming::category = "Standard";
const char* NSGConstantQStreaming::description = DOC("This algorithm computes a constant Q transform using non stationary Gabor frames and returns a complex time-frequency representation of the input vector.\n"
"The implementation is inspired by the toolbox described in [1]."
"\n"
"References:\n"
"  [1] Schörkhuber, C., Klapuri, A., Holighaus, N., & Dörfler, M. (n.d.). A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.");

NSGConstantQStreaming::NSGConstantQStreaming() : Algorithm() {

  declareInput(_frame, "frame", "the input audio signal");
  declareOutput(_constantQ, "constantq", "the constant Q transform of the input frame");
  declareOutput(_constantQDC, "constantqdc", "the DC band transform of the input frame. Only needed for the inverse transform");
  declareOutput(_constantQNF, "constantqnf", "the Nyquist band transform of the input frame. Only needed for the inverse transform");
  declareOutput(_frameStamps, "framestamps", "this vector sets the beginnings of each frame in the 'constantq' buffer");

  _wrapper =  AlgorithmFactory::create("NSGConstantQ");

  _frame >> _wrapper->input("frame");

  _wrapper->output("constantq")   >> _constantQinner;
  _wrapper->output("constantqdc") >> _constantQDCinner;
  _wrapper->output("constantqnf") >> _constantQNFinner;
}

void NSGConstantQStreaming::configure() {

  std::string rasterize = parameter("rasterize").toLower();

  if (rasterize != "full") {
    throw EssentiaException("NSGConstantQStreaming: This algorithm was designed to work only with 'rasterize' = 'full'");
  }
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

  _constantQinner.setAcquireSize(1);
  _constantQinner.setReleaseSize(1);

  _constantQDCinner.setAcquireSize(1);
  _constantQDCinner.setReleaseSize(1);

  _constantQNFinner.setAcquireSize(1);
  _constantQNFinner.setReleaseSize(1);

  // @todo this is a workaround to initialize the number of CQ time-stamps
  // to push to a high value. It would be better to compute the actual
  // number from the input parameters.

  _constantQ.setAcquireSize(200);
  _constantQ.setReleaseSize(200);

  _constantQDC.setAcquireSize(1);
  _constantQDC.setReleaseSize(1);

  _constantQNF.setAcquireSize(1);
  _constantQNF.setReleaseSize(1);

  _frameStamps.setAcquireSize(1);
  _frameStamps.setReleaseSize(1);

  _frameStampsCount = 0;
}

AlgorithmStatus NSGConstantQStreaming::process() {
  _wrapper->process();

  bool ok = _constantQinner.acquire(1);
  _constantQDCinner.acquire(1);
  _constantQNFinner.acquire(1);


  if ( !ok ) return NO_INPUT;

  // @todo another workaround! This tries to prevent the connected algorithms to crash.
  // As we are releasing 'timeStamps' tokens for each input token, there are too many to
  // process in the input data of the upcoming algorithms once shouldStop() is enabled.
  // Thus, here it just discards the remaining tokens. Tested with CartesianToPoolar().
  if( shouldStop() ) return FINISHED;

  const std::vector<vector<std::vector<std::complex<Real> > > > &constantQ = _constantQinner.tokens();
  const std::vector<std::vector<std::complex<Real> > >  &constantQdc = _constantQDCinner.tokens();
  const std::vector<std::vector<std::complex<Real> > >  &constantQnf = _constantQNFinner.tokens();

  unsigned timeStamps = constantQ[0][0].size();
  unsigned channSize = constantQ[0].size();

  _constantQ.setAcquireSize(timeStamps);
  _constantQ.setReleaseSize(timeStamps);


  _constantQ.acquire(timeStamps);
  _constantQDC.acquire();
  _constantQNF.acquire();
  _frameStamps.acquire();


  std::vector<std::vector<std::complex<Real> > >& constantQout = _constantQ.tokens();
  std::vector<std::vector<std::complex<Real> > >& constantQDCout = _constantQDC.tokens();
  std::vector<std::vector<std::complex<Real> > >& constantQNFout = _constantQNF.tokens();
  std::vector<int>& frameStamps = _frameStamps.tokens();

  constantQDCout = constantQdc;
  constantQNFout = constantQnf;

  frameStamps[0] = _frameStampsCount;
  _frameStampsCount += timeStamps;

  std::vector<std::complex<float> > item;
  for (unsigned i=0; i<timeStamps; i++){
    for (unsigned j=0; j<channSize; j++) item.push_back(constantQ[0][j][i]);

    constantQout[i] = item;
    item.clear();
  }


  EXEC_DEBUG("releasing");
  _constantQinner.release();
  _constantQDCinner.release();
  _constantQNFinner.release();

  _constantQ.release(timeStamps);
  _constantQDC.release();
  _constantQNF.release();

  _frameStamps.release();

  EXEC_DEBUG("released");

  return OK;
}


}
}
