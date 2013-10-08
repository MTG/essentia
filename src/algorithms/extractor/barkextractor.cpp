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

#include "barkextractor.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "copy.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const char* BarkExtractor::name = "BarkExtractor";
const char* BarkExtractor::description = DOC("This algorithm extracts some Bark bands based spectral features from an audio signal");

BarkExtractor::BarkExtractor() : _configured(false) {
  // input:
  declareInput(_signal, "signal", "the input audio signal");

  // outputs:
  declareOutput(_bbands, "barkbands", "spectral energy at each bark band. See BarkBands alogithm");
  declareOutput(_bbandsKurtosis, "barkbands_kurtosis", "kurtosis from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_bbandsSkewness, "barkbands_skewness", "skewness from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_bbandsSpread, "barkbands_spread", "spread from barkbands. See DistributionShape algorithm documentation");
  declareOutput(_crestValue, "spectral_crest", "See Crest algorithm documentation");
  declareOutput(_flatness, "spectral_flatness_db", "See flatnessDB algorithm documentation");


  // create network (instantiate algorithms)
  createInnerNetwork();

  // wire all this up!
  _signal                              >>  _frameCutter->input("signal");

  // spectrum
  _frameCutter->output("frame")        >>  _windowing->input("frame");
  _windowing->output("frame")          >>  _spectrum->input("frame");

  // barkbands
  _spectrum->output("spectrum")               >>  _barkBands->input("spectrum");

  _barkBands->output("bands")                 >>  _bbands;

  _barkBands->output("bands")                 >>  _crest->input("array");
  _crest->output("crest")                     >>  _crestValue;

  _barkBands->output("bands")                 >>  _flatnessdb->input("array");
  _flatnessdb->output("flatnessDB")           >>  _flatness;

  _barkBands->output("bands")                 >>  _centralMoments->input("array");
  _centralMoments->output("centralMoments")   >>  _distributionShape->input("centralMoments");
  _distributionShape->output("kurtosis")      >>  _bbandsKurtosis;
  _distributionShape->output("skewness")      >>  _bbandsSkewness;
  _distributionShape->output("spread")        >>  _bbandsSpread;

  _network = new scheduler::Network(_frameCutter);
}

void BarkExtractor::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _barkBands          = factory.create("BarkBands",
                                       "numberBands", 27);
  _centralMoments     = factory.create("CentralMoments",
                                       "range", 26);
  _crest              = factory.create("Crest");
  _distributionShape  = factory.create("DistributionShape");
  _flatnessdb         = factory.create("FlatnessDB");
  _frameCutter        = factory.create("FrameCutter");
  _spectrum           = factory.create("Spectrum");
  _windowing          = factory.create("Windowing",
                                       "type", "blackmanharris62");
}

void BarkExtractor::configure() {
  int frameSize   = parameter("frameSize").toInt();
  int hopSize     = parameter("hopSize").toInt();

  _barkBands->configure(INHERIT("sampleRate"));
  _frameCutter->configure("silentFrames", "noise", "hopSize", hopSize, "frameSize", frameSize);
}

BarkExtractor::~BarkExtractor() {
  clearAlgos();
}

void BarkExtractor::clearAlgos() {
  if (!_configured) return;
  delete _network;
}
