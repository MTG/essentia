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

#include <streaming_extractorpanning.h>
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>

using namespace std;
using namespace essentia;
using namespace streaming;

void Panning(SourceBase& input, Pool& pool, const Pool& options, const string& nspace) {

   // namespace
  string llspace = "lowlevel.";
  if (!nspace.empty()) llspace = nspace + ".lowlevel.";

  Real sampleRate = options.value<Real>("sampleRate");
  int frameSize   = int(options.value<Real>("panning.frameSize"));
  int hopSize     = int(options.value<Real>("panning.hopSize"));
  int zeroPadding = int(options.value<Real>("panning.zeroPadding"));
  string silentFrames = options.value<string>("panning.silentFrames");
  string windowType = options.value<string>("panning.windowType");

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* demuxer = factory.create("StereoDemuxer");

  Algorithm* fc_left = factory.create("FrameCutter",
                                      "frameSize", frameSize,
                                      "hopSize", hopSize,
                                      "startFromZero", false,
                                      "silentFrames", silentFrames);

  Algorithm* fc_right  = factory.create("FrameCutter",
                                        "frameSize", frameSize,
                                        "hopSize", hopSize,
                                        "startFromZero", false,
                                        "silentFrames", silentFrames);

  Algorithm* w_left= factory.create("Windowing",
                                    "size", frameSize,
                                    "zeroPadding", zeroPadding,
                                    "type", windowType);

  Algorithm* w_right = factory.create("Windowing",
                                      "size", frameSize,
                                      "zeroPadding", zeroPadding,
                                      "type", windowType);

  Algorithm* spec_left = factory.create("Spectrum");

  Algorithm* spec_right = factory.create("Spectrum");

  Algorithm* pan = factory.create("Panning",
                                  "sampleRate", sampleRate,
                                  "averageFrames", 43, // 2 seconds * sr/hopsize
                                  "panningBins", 512,
                                  "numCoeffs", 20,
                                  "numBands", 1,
                                  "warpedPanorama", true);

  connect(input, demuxer->input("audio"));
  connect(demuxer->output("left"), fc_left->input("signal"));
  connect(demuxer->output("right"), fc_right->input("signal"));
  // left channel
  connect(fc_left->output("frame"), w_left->input("frame"));
  connect(w_left->output("frame"), spec_left->input("frame"));
  connect(spec_left->output("spectrum"), pan->input("spectrumLeft"));
  // right channel
  connect(fc_right->output("frame"), w_right->input("frame"));
  connect(w_right->output("frame"), spec_right->input("frame"));
  connect(spec_right->output("spectrum"), pan->input("spectrumRight"));

  // panning:
  connect(pan->output("panningCoeffs"), pool, llspace + "panning_coefficients");
}
