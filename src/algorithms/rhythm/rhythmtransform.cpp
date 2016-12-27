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

#include "rhythmtransform.h"
#include "essentiamath.h"
#include "tnt/tnt2vector.h"
#include <cfloat>
using namespace std;

namespace essentia {
namespace standard {

const char* RhythmTransform::name = "RhythmTransform";
const char* RhythmTransform::category = "Rhythm";
const char* RhythmTransform::description = DOC("This algorithm implements the rhythm transform. It computes a tempogram, a representation of rhythmic periodicities in the input signal in the rhythm domain, by using FFT similarly to computation of spectrum in the frequency domain [1]. Additional features, including rhythmic centroid and a rhythmic counterpart of MFCCs, can be derived from this rhythmic representation.\n\n"
"The algorithm relies on a time sequence of frames of Mel bands energies as an input (see MelBands), but other types of frequency bands can be used as well (see BarkBands, ERBBands, FrequencyBands). For each band, the derivative of the frame to frame energy evolution is computed, and the periodicity of the resulting signal is computed: the signal is cut into frames of \"frameSize\" size and is analyzed with FFT. For each frame, the obtained power spectrums are summed across all bands forming a frame of rhythm transform values.\n"
"\n"
"Quality: experimental (non-reliable, poor accuracy according to tests on simple loops, more tests are necessary)\n"
"\n"
"References:\n"
"  [1] E. Guaus and P. Herrera, \"The rhythm transform: towards a generic\n"
"  rhythm description,\" in International Computer Music Conference (ICMC’05),\n"
"  2005.");


void RhythmTransform::configure() {
  _rtFrameSize = parameter("frameSize").toInt();
  _rtHopSize = parameter("hopSize").toInt();
}

void RhythmTransform::compute() {
  const vector<vector<Real> >& bands = _melBands.get();
  vector<vector<Real> >& output = _rhythmTransform.get();

  int nFrames = bands.size();
  int nBands = bands[0].size();

  // derive and transpose
  vector<vector<Real> > bandsDerivative(nBands);
  for (int band=0; band<nBands; ++band) {
    vector<Real> derivTemp(nFrames);
    derivTemp[0] = 0.0;
    for (int frame=1; frame<nFrames; frame++) {
      derivTemp[frame] = bands[frame][band]-bands[frame-1][band];
    }
    bandsDerivative[band] = derivTemp;
  }

  int i = 0;
  // in the original implementation, computation was stopped at:
  // (i+_rtFrameSize<nFrames). However, there might be quite a lot of the
  // signal not being analyzed. Therefore the new implementation computes the
  // whole signal and zero pads if it i+rtFrameSize exceeds the number of
  // melbands frames
  while (i<nFrames) {
    vector<Real> bandSpectrum(_rtFrameSize/2+1);
    for (int band=0; band<nBands; band++) {
      vector<Real> rhythmFrame(_rtFrameSize);
      for (int j=0; j<_rtFrameSize; ++j) {
        if (i+j<nFrames){
	        rhythmFrame[j] = bandsDerivative[band][i+j];
        }
        else rhythmFrame[j] = 0.0; // zeropadding
      }
      vector<Real> windowedFrame;
      vector<Real> rhythmSpectrum;

      _w->input("frame").set(rhythmFrame);
      _w->output("frame").set(windowedFrame);
      _spec->input("frame").set(windowedFrame);
      _spec->output("spectrum").set(rhythmSpectrum);

      _w->compute();
      _spec->compute();

      // square the resulting spectrum, sum periodograms across bands
      for (int bin=0; bin<(int)rhythmSpectrum.size(); ++bin)
	      bandSpectrum[bin] += rhythmSpectrum[bin]*rhythmSpectrum[bin];
    }
    output.push_back(bandSpectrum);
    i += _rtHopSize;
  }
  // skip the last couple of frames as they don't make up
  // a full frame consisting of rmsSize rms values.
}

} // namespace standard
} // namespace streaming

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* RhythmTransform::name = standard::RhythmTransform::name;
const char* RhythmTransform::description = standard::RhythmTransform::description;

RhythmTransform::RhythmTransform() : AlgorithmComposite() {

  _poolStorage = new PoolStorage<vector<Real> >(&_pool, "internal.mel_bands");
  _rhythmAlgo = standard::AlgorithmFactory::create("RhythmTransform");

  declareInput(_poolStorage->input("data"), 1, "melBands","the energy in the melbands");
  declareOutput(_rhythmTransform, 0, "rhythm", "consecutive frames in the rhythm domain");
  _rhythmTransform.setBufferType(BufferUsage::forMultipleFrames);
}

void RhythmTransform::configure() {
  _rhythmAlgo->configure(INHERIT("frameSize"),
                         INHERIT("hopSize"));
}

RhythmTransform::~RhythmTransform() {
  delete _rhythmAlgo;
  delete _poolStorage;
}

AlgorithmStatus RhythmTransform::process() {
  if (!shouldStop()) return PASS;

  const vector<vector<Real> >& bands = _pool.value<vector<vector<Real> > >("internal.mel_bands");
  vector<vector<Real> > rhythmTransform;

  _rhythmAlgo->input("melBands").set(bands);
  _rhythmAlgo->output("rhythm").set(rhythmTransform);
  _rhythmAlgo->compute();

  _rhythmTransform.push(vecvecToArray2D(rhythmTransform));

  return OK;
}

void RhythmTransform::reset() {
  AlgorithmComposite::reset();
  _rhythmAlgo->reset();
  _pool.clear();
}

} // namespace streaming
} // namespace essentia
