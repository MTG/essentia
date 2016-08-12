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

#include "superfluxextractor.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "copy.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const char* SuperFluxExtractor::name = "SuperFluxExtractor";
const char* SuperFluxExtractor::category = "Rhythm";
const char* SuperFluxExtractor::description = DOC(
"This algorithm detects onsets given an audio signal using SuperFlux algorithm [1]. This implementation is based on the available reference implementation in python [2].\n"
"\n"
"References:\n"
"  [1] Böck, S. and Widmer, G., Maximum Filter Vibrato Suppression for Onset\n"
"  Detection, Proceedings of the 16th International Conference on Digital Audio\n"
"  Effects (DAFx-13), 2013\n"
"  [2] https://github.com/CPJKU/SuperFlux");

SuperFluxExtractor::SuperFluxExtractor() : _configured(false) {

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_onsets, "onsets","lists of onsets");

  // create network (instantiate algorithms)
  createInnerNetwork();

  // wire all this up!
  _signal                            >> _frameCutter->input("signal");
  _frameCutter->output("frame")      >> _w->input("frame");
  _w->output("frame")                >> _spectrum->input("frame");
  _spectrum->output("spectrum")      >> _triF->input("spectrum");
  _triF->output("bands")             >> _superFluxF->input("bands");
  _superFluxF->output("differences") >> _superFluxP->input("novelty");
  _superFluxP->output("peaks")       >> _onsets;

  _network = new scheduler::Network(_frameCutter);
}

void SuperFluxExtractor::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
	
  _frameCutter = factory.create("FrameCutter");
  _w = factory.create("Windowing", "type", "hann");
  _spectrum = factory.create("Spectrum");
  _triF = factory.create("TriangularBands", "log", false);
  _superFluxP = factory.create("SuperFluxPeaks");
  _superFluxF = factory.create("SuperFluxNovelty", "binWidth", 8, "frameWidth", 2);
    
  _vout = new essentia::streaming::VectorOutput<Real>();
}

void SuperFluxExtractor::configure() {
  int frameSize   = parameter("frameSize").toInt();
  int hopSize     = parameter("hopSize").toInt();
  Real sampleRate = parameter("sampleRate").toReal();
  
	_frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "startFromZero", false,
                          "validFrameThresholdRatio", 0,
                          "lastFrameToEndOfFile", false,
                        	"silentFrames", "keep");

  _superFluxP->configure(INHERIT("ratioThreshold"), 
                        INHERIT("threshold"),
                        "frameRate", sampleRate/hopSize,
                        INHERIT("combine"),
                        "pre_avg", 100.,
                        "pre_max", 30.);
}

SuperFluxExtractor::~SuperFluxExtractor() {
  clearAlgos();
}

void SuperFluxExtractor::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


namespace essentia {
namespace standard {

const char* SuperFluxExtractor::name = "SuperFluxExtractor";
const char* SuperFluxExtractor::category = "";
const char* SuperFluxExtractor::description = streaming::SuperFluxExtractor::description;

SuperFluxExtractor::SuperFluxExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_onsets, "onsets", "the onsets times");
  createInnerNetwork();
}

SuperFluxExtractor::~SuperFluxExtractor() {
  delete _network;
}

void SuperFluxExtractor::reset() {
  _network->reset();
}

void SuperFluxExtractor::configure() {
  _SuperFluxExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"),INHERIT("sampleRate"),INHERIT("threshold"),INHERIT("combine"),INHERIT("ratioThreshold"));
}

void SuperFluxExtractor::createInnerNetwork() {
  _SuperFluxExtractor = streaming::AlgorithmFactory::create("SuperFluxExtractor");
  _vectorInput = new streaming::VectorInput<Real>();
  _vectorOut = new streaming::VectorOutput<std::vector<Real> >();
  
  *_vectorInput >> _SuperFluxExtractor->input("signal");
  _SuperFluxExtractor->output("onsets") >> _vectorOut->input("data"); //PC(_pool, "onsets.times");
  _network = new scheduler::Network(_vectorInput);
}


 void SuperFluxExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  vector<Real>& onsets = _onsets.get();

  vector<vector<Real> > ll;
  _vectorInput->setVector(&signal);
  _vectorOut->setVector(&ll);
  _network->run(); 
  onsets = ll[0]; // FIXME will this ever fail?
}

} // namespace standard
} // namespace essentia
