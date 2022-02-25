/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"
#include "essentia/utils/tnt/tnt2vector.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;


int main(int argc, char* argv[]) {

  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " <query_audio_file> <reference_audio_file> <json_output_path>" << endl;
    creditLibAV();
    exit(1);
  }

  string queryFilename = argv[1];
  string referenceFilename = argv[2];
  string outputFilename = argv[3];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;
  int frameSize = 4096;
  int hopSize = 2048;
  int numBins = 12;
  Real minFrequency = 100;
  Real maxFrequency = 3500;

  // we want to compute the HPCP of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> Spectrum -> SpectralPeaks -> SpectralWhitening -> HPCP
  // Later we want compute the cover song similarity score from the input HPCP features.
  // HPCP -> CrossSimilarityMatrix -> CoverSongSimilarity

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();


  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", queryFilename,
                                    "sampleRate", sampleRate);
  

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");

  Algorithm* peak = factory.create("SpectralPeaks",
                                   "sampleRate", sampleRate);

  Algorithm* white = factory.create("SpectralWhitening",
                                    "maxFrequency", maxFrequency,
                                    "sampleRate", sampleRate);

  Algorithm* hpcp = factory.create("HPCP",
                                   "sampleRate", sampleRate,
                                   "minFrequency", minFrequency,
                                   "maxFrequency", maxFrequency,
                                   "size", numBins);

  // with default params
  Algorithm* csm = factory.create("ChromaCrossSimilarity");

  Algorithm* coversim = factory.create("CoverSongSimilarity",
                                      "alignmentType", "chen17"); 

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos for hpcp, csm and local-alignment extraction ---------" << endl;

  vector<Real> audioBuffer;

  // audio -> FrameCutter
  audio->output("audio").set(audioBuffer);
  fc->input("signal").set(audioBuffer);

  // FrameCutter -> Windowing -> Spectrum
  vector<Real> frame, windowedFrame;

  fc->output("frame").set(frame);
  w->input("frame").set(frame);

  w->output("frame").set(windowedFrame);
  spec->input("frame").set(windowedFrame);

  vector<Real> spect;
  spec->output("spectrum").set(spect);

  // Spectrum -> SpectralPeaks -> SpectralWhitening
  vector<Real> peakFrequencies, peakMagnitudes;
  peak->input("spectrum").set(spect);
  peak->output("frequencies").set(peakFrequencies);
  peak->output("magnitudes").set(peakMagnitudes);

  vector<Real> wPeakMagnitudes;
  white->input("spectrum").set(spect);
  white->input("frequencies").set(peakFrequencies);
  white->input("magnitudes").set(peakMagnitudes);
  white->output("magnitudes").set(wPeakMagnitudes);

  // SpectralWhitening > HPCP
  vector<Real> hpcpOut;
  hpcp->input("frequencies").set(peakFrequencies);
  hpcp->input("magnitudes").set(wPeakMagnitudes);
  hpcp->output("hpcp").set(hpcpOut);

  // TODO: replace with std::vector<vector<Real> > when essentia pool has 2D vector support
  // TNT::Array2D<Real> hpcpOutPool;

  /////////// STARTING THE ALGORITHMS //////////////////

  // compute HPCP feature of the query song
  cout << "-------- start processing hpcp for " << queryFilename << " --------" << endl;

  audio->compute();

  while (true) {

    // compute a frame
    fc->compute();
    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }
    // if the frame is silent, just drop it and go on processing
    if (isSilent(frame)) continue;

    w->compute();
    spec->compute();
    peak->compute();
    white->compute();
    hpcp->compute();
    pool.add("queryHPCP", hpcpOut);
  }

  // Now we reset the audio loader and compute HPCP feature for the reference song
  cout << "-------- start processing hpcp for " << referenceFilename << " --------" << endl;
  audioBuffer.clear();
  hpcpOut.clear();
  audio->reset();
  audio->configure("filename", referenceFilename,
                   "sampleRate", sampleRate);
  fc->configure("frameSize", frameSize,
                "hopSize", hopSize);

  audio->compute();

  while (true) {

    // compute a frame
    fc->compute();
    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }
    // if the frame is silent, just drop it and go on processing
    if (isSilent(frame)) continue;

    w->compute();
    spec->compute();
    peak->compute();
    white->compute();
    hpcp->compute();
    pool.add("referenceHPCP", hpcpOut);

  }

  const vector<vector<Real> > queryHpcp = pool.value<vector<vector<Real> > >("queryHPCP");
  const vector<vector<Real> > referenceHpcp = pool.value<vector<vector<Real> > >("referenceHPCP");
  cout << "Query HPCP frames: " << queryHpcp.size() << "\nReference HPCP frames: " << referenceHpcp.size() << endl; 

  /////////// CONNECTING THE ALGORITHMS FOR COVER SONG SIMILARITY ////////////////
  cout << "\n-------- computing cover song similarity ---------" << endl;

  vector<vector<Real> > simMatrix;
  Real distance;
  csm->input("queryFeature").set(queryHpcp);
  csm->input("referenceFeature").set(referenceHpcp);
  csm->output("csm").set(simMatrix);

  vector<vector<Real> > scoreMatrix;
  coversim->input("inputArray").set(simMatrix);
  coversim->output("scoreMatrix").set(scoreMatrix);
  coversim->output("distance").set(distance);

  // Now we compute the cover song similarity
  csm->compute();

  cout << " .... computing smith-waterman local alignment" << endl;
  coversim->compute();
  // TODO: replace with std::vector<vector<Real> > when essentia pool has 2D vector support
  cout << "Cover song similarity distance: " << distance << endl;
  pool.add("distance", distance);
  pool.add("scoreMatrix", vecvecToArray2D(scoreMatrix));
  pool.remove("queryHPCP");
  pool.remove("referenceHPCP");
    
  // write results to file
  cout << "\n-------- writing results to file " << outputFilename << " ---------" << endl;
  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename,
                                               "format", "json");
  output->input("pool").set(pool);
  output->compute();

  delete audio;
  delete fc;
  delete w;
  delete spec;
  delete peak;
  delete white;
  delete hpcp;
  delete csm;
  delete coversim;
  delete output;

  essentia::shutdown();

  return 0;
}

