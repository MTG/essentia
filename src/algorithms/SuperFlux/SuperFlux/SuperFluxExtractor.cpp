/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "SuperFluxextractor.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "copy.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const char* SuperFluxExtractor::name = "SuperFluxExtractor";
const char* SuperFluxExtractor::description = DOC("This algorithm extracts some Bark bands based spectral features from an audio signal");

SuperFluxExtractor::SuperFluxExtractor() : _configured(false) {
  // input:
  declareInput(_signal, "signal", "the input audio signal");

  // outputs:
  declareOutput(_onsets, "onsets","lists of onsets");



  // create network (instantiate algorithms)
  createInnerNetwork();

  // wire all this up!
  _signal                             >>  _frameCutter->input("signal");

    _frameCutter->output("frame")  >>  w->input("frame");
    w->output("frame") >> spectrum->input("frame");
    spectrum->output("spectrum") >> triF->input("spectrum");
    triF->output("bands")>>superFluxF->input("bands");
    superFluxF->output("Differences")  >>superFluxP->input("novelty");
    superFluxP->output("peaks") >> _onsets;
    
    

  _network = new scheduler::Network(_frameCutter);
}

void SuperFluxExtractor::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
	_frameCutter = factory.create(	"FrameCutter");
    
    w = factory.create("Windowing","type","hann");
    
    spectrum = factory.create("Spectrum");
    triF = factory.create("Triangularbands","Log",true);
    superFluxP = factory.create("SuperFluxPeaks");
    superFluxF = factory.create("SuperFluxNovelty","binWidth",3,"frameWidth",2);
    
    vout = new essentia::streaming::VectorOutput<Real>();
   
}

void SuperFluxExtractor::configure() {
  int frameSize   = parameter("frameSize").toInt();
  int hopSize     = parameter("hopSize").toInt();
  Real threshold = parameter("threshold").toReal();
  Real sampleRate = parameter("sampleRate").toReal();
  Real combine = parameter("combine").toReal();


  
	_frameCutter->configure(
                        			"frameSize",frameSize,
                        			"hopSize",hopSize,
                        			"startFromZero" , true,
                        			"validFrameThresholdRatio", 1,
                        			"lastFrameToEndOfFile",true,
                        			"silentFrames","keep"
                        		);


  superFluxP->configure("rawmode" , false,"threshold" ,threshold/NOVELTY_MULT,"startFromZero",true,"frameRate", sampleRate*1.0/hopSize,"combine",combine);


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
const char* SuperFluxExtractor::description = DOC("retreive onsets times implementing superflux algorithm");

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
  _pool.clear();
}

void SuperFluxExtractor::configure() {
  _SuperFluxExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"),INHERIT("sampleRate"),INHERIT("threshold"),INHERIT("combine"));
}

void SuperFluxExtractor::createInnerNetwork() {
  _SuperFluxExtractor = streaming::AlgorithmFactory::create("SuperFluxExtractor");
  _vectorInput = new streaming::VectorInput<Real>();
    _vectorOut = new streaming::VectorOutput<std::vector<Real> >();

  *_vectorInput                        >>  _SuperFluxExtractor->input("signal");
  _SuperFluxExtractor->output("onsets")  >>  _vectorOut->input("data");//PC(_pool, "onsets.times");


  _network = new scheduler::Network(_vectorInput);
}


 void SuperFluxExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);
     vector<Real>& onsets = _onsets.get();
     vector<vector<Real> > ll;
     _vectorOut->setVector(&ll);
     _network->run();
     
     onsets = ll[0];
  

  //onsets = _pool.value<vector<Real> >("onsets.times");
}

} // namespace standard
} // namespace essentia
