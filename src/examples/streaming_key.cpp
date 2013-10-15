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

#include <iostream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

Real ReplayGain(const string& filename, Pool& pool) {
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio       = factory.create("EqloudLoader",
                                          "filename", filename,
                                          "sampleRate", 44100,
                                          "downmix", "mix");

  Algorithm* replay_gain = factory.create("ReplayGain", "applyEqloud", false);

  audio->output("audio")             >>  replay_gain->input("signal");
  replay_gain->output("replayGain")  >>  PC(pool, "metadata.audio_properties.replay_gain");

  Network(audio).run();

  return pool.value<Real>("metadata.audio_properties.replay_gain");
}

void TuningFrequency(const string& filename,
                     int framesize, int hopsize, int zeropadding,
                     Real rgain, Pool& pool) {
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio         = factory.create("EqloudLoader",
                                            "filename", filename,
                                            "sampleRate", 44100,
                                            "replayGain", rgain,
                                            "downmix", "mix");

  Algorithm* frameCutter   = factory.create("FrameCutter",
                                            "frameSize", framesize,
                                            "hopSize", hopsize,
                                            "silentFrames", "noise",
                                            "startFromZero", true);

  Algorithm* window        = factory.create("Windowing", "type", "blackmanharris62");

  Algorithm* spectrum      = factory.create("Spectrum", "size", framesize + zeropadding);

  Algorithm* spectralPeaks = factory.create("SpectralPeaks",
                                            "sampleRate", 44100,
                                            "maxPeaks", 40,
                                            "maxFrequency", 5000.,
                                            "minFrequency", 40.,
                                            "magnitudeThreshold", 0.00001,
                                            "orderBy", "magnitude");

  Algorithm* tuningFreq    = factory.create("TuningFrequency", "resolution", 1.0);

  // make connectinons:
  audio->output("audio")                 >>  frameCutter->input("signal");
  frameCutter->output("frame")           >>  window->input("frame");
  window->output("frame")                >>  spectrum->input("frame");
  spectrum->output("spectrum")           >>  spectralPeaks->input("spectrum");
  spectralPeaks->output("frequencies")   >>  tuningFreq->input("frequencies");
  spectralPeaks->output("magnitudes")    >>  tuningFreq->input("magnitudes");
  tuningFreq->output("tuningFrequency")  >>  PC(pool, "tonal.tuning_freq");
  tuningFreq->output("tuningCents")      >>  PC(pool, "tonal.tuning_cents");

  Network(audio).run();
}

void TonalDescriptors(const string& filename,
                      int framesize, int hopsize, int zeropadding,
                      Real rgain, Real tuningFrequency, Pool& pool) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio         = factory.create("EqloudLoader",
                                            "filename", filename,
                                            "sampleRate", 44100,
                                            "replayGain", rgain,
                                            "downmix", "mix");

  Algorithm* frameCutter   = factory.create("FrameCutter",
                                            "frameSize", framesize,
                                            "hopSize", hopsize,
                                            "silentFrames", "noise",
                                            "startFromZero", true);

  Algorithm* window        = factory.create("Windowing", "type", "hann");

  Algorithm* spectrum      = factory.create("Spectrum", "size", framesize + zeropadding);

  Algorithm* spectralPeaks = factory.create("SpectralPeaks",
                                            "sampleRate", 44100,
                                            "maxPeaks", 40,
                                            "maxFrequency", 5000,
                                            "minFrequency", 40,
                                            "magnitudeThreshold", 0.00001,
                                            "orderBy", "frequency");


  Algorithm* key           = factory.create("Key");
  // key is not configurable in streaming mode!!

  Algorithm* hpcp          = factory.create("HPCP",
                                            "size", 36,
                                            "referenceFrequency", tuningFrequency,
                                            "bandPreset", false,
                                            "minFrequency", 40.,
                                            "maxFrequency", 5000.,
                                            "weightType", "squaredCosine",
                                            "nonLinear", false,
                                            "windowSize", 4./3.,
                                            "sampleRate", 44100);

  // make connectinons:
  audio->output("audio")                >>  frameCutter->input("signal");
  frameCutter->output("frame")          >>  window->input("frame");
  window->output("frame")               >>  spectrum->input("frame");
  spectrum->output("spectrum")          >>  spectralPeaks->input("spectrum");
  spectralPeaks->output("frequencies")  >>  hpcp->input("frequencies");
  spectralPeaks->output("magnitudes")   >>  hpcp->input("magnitudes");
  hpcp->output("hpcp")                  >>  key->input("pcp");

  // data storage:
  // yaml file:
  key->output("key")                    >>  PC(pool, "tonal.key");
  key->output("scale")                  >>  PC(pool, "tonal.key_scale");
  key->output("strength")               >>  PC(pool, "tonal.key_strength");

  Network(audio).run();
}

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile" << endl;
    exit(1);
  }

  string filename = argv[1];

  // parameters:
  uint framesize = 4096;
  uint hopsize = 2048;
  uint zeropadding = 0;

  Real tuningFrequency = 440.;

  essentia::init();

  // data storage:
  Pool pool;
  // configure the pool to display the statistics we want:
  vector<string> stats;
  vector<string> exceptions;
  stats.push_back("mean");
  exceptions.push_back("mean");
  stats.push_back("min");
  stats.push_back("max");
  map<string, vector<string> > except;
  except["tonal.tuning_cents"] = exceptions;;

  //cout << "computing replay gain:\t";
  Real rgain = ReplayGain(filename, pool);
  //cout << rgain << endl;

  cout << "tuning frequency:\t";
  // compute tuning frequency and store it in the pool:
  TuningFrequency(filename, framesize, hopsize, zeropadding, rgain,  pool);
  // retrieve the mean tuning frequency from the pool:
  tuningFrequency = mean(pool.value<vector<Real> >("tonal.tuning_freq"));
  cout << tuningFrequency << endl;

  cout << "key: ";
  TonalDescriptors(filename, framesize, hopsize, zeropadding,  rgain, tuningFrequency, pool);
  cout << "\t" << pool.value<string>("tonal.key")
       << "  " << pool.value<string>("tonal.key_scale")<< endl;

  // write to yaml file:
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", argv[2]);

  standard::Algorithm* aggregator = standard::AlgorithmFactory::create("PoolAggregator",
                                                                       "exceptions", except,
                                                                       "defaultStats", stats);
  Pool poolStats;

  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);
  output->input("pool").set(poolStats);
  output->compute();

  delete output;

  essentia::shutdown();

  return 0;
}
