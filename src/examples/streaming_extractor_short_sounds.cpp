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
#include <sstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiautil.h>
#include <essentia/scheduler/network.h>

// helper functions
#include "streaming_extractorutils.h"
#include "streaming_extractorlowlevel.h"
#include "streaming_extractorsfx.h"
#include "streaming_extractortonal.h"
#include "streaming_extractorpanning.h"
#include "streaming_extractorpostprocess.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

void computeSegments(const string& audioFilename, Real startTime, Real endTime,
                     Pool& pool, const Pool& options);
void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& pool, const Pool& options);

void computeReplayGain(const string& audioFilename, Real startTime, Real endTime,
                       Pool& pool, const Pool& options);
void computeLowLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& pool, const Pool& options, const string& nspace = "");
void computeMidLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& pool, const Pool& options, const string& nspace = "");
void computePanning(const string& audioFilename, Real startTime, Real endTime,
                    Pool& pool, const Pool& options, const string& nspace = "");
void computeHighlevel(Pool& pool, const Pool& options, const string& nspace = "");
Pool computeAggregation(Pool& pool, int segments=0);
void outputToFile(Pool& pool, const string& outputFilename);

void usage() {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: streaming_extractor input_audiofile output_textfile [segment=true|false] | [ startTime endTime ]" << endl;
    exit(1);
}

int main(int argc, char* argv[]) {

  //if (argc < 3  || argc > 6) usage();
  string audioFilename;
  string outputFilename;
  bool computeSegmentation = false;
  Real startTime = 0, endTime = 2000;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    case 4:
    {
      audioFilename =  argv[1];
      outputFilename = argv[2];
      string s(argv[3]);
      string::size_type pos = s.find("segment=");
      if (pos == string::npos) {
        cout << "Unrecognize option \"" << argv[3] << "\"" << endl;
        exit(2);
      }
      string segmentation = s.substr(s.find("=")+1, string::npos);
      if (segmentation=="true") computeSegmentation = true;
      break;
    }
    case 5:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      startTime = atof(argv[3]);
      endTime = atof(argv[4]);
      cout << "processing audio from " << startTime << "s. to " << endTime << "s." << endl;
      break;
    case 6:
      cout << "Segmentation only available on the entire audio file" << endl;
      break;
    default:
      usage();
  }

  // Register the algorithms in the factory(ies)
  essentia::init();

  // pool cotaining profile (configure) options; use default settings
  Pool options;
  setOptions(options, "");

  // pool for storing results
  Pool pool;
  // this pool contains only descriptors computed after applying equal loudness
  // filetering:
  pool.set("metadata.audio_properties.equal_loudness", true);

  if (computeSegmentation) {
    // pool for storing segments:
    computeReplayGain(audioFilename, startTime, endTime, pool, options);
    computeSegments(audioFilename, startTime, endTime, pool, options);
    vector<Real> segments = pool.value<vector<Real> >("segmentation.timestamps");
    for (int i=0; i<int(segments.size()-1); ++i) {
      startTime = segments[i];
      endTime = segments[i+1];
      cout << "\n**************************************************************************";
      cout << "\nSegment " << i << ": processing audio from " << startTime << "s. to " << endTime << "s.";
      cout << "\n**************************************************************************" << endl;
      ostringstream ns;
      ns << "segment_" << i ;
      computeLowLevel(audioFilename, startTime, endTime, pool, options, ns.str());
      computeMidLevel(audioFilename, startTime, endTime, pool, options, ns.str());
      //computePanning(audioFilename, startTime, endTime, pool, options, ns.str());
      computeHighlevel(pool, options, ns.str());
    }
    cout << "\n**************************************************************************" << endl;
    Pool stats = computeAggregation(pool, segments.size());
    outputToFile(stats, outputFilename);
  }
  else {
    try {
      compute(audioFilename, outputFilename, startTime, endTime, pool, options);
    }
    catch (EssentiaException& e) {
      cout << e.what() << endl;
    }
  }

  pool.remove("metadata.audio_properties.downmix");
  essentia::shutdown();

  return 0;
}

void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& pool, const Pool& options) {
  computeReplayGain(audioFilename, startTime, endTime, pool, options);
  computeLowLevel(audioFilename, startTime, endTime, pool, options);
  computeMidLevel(audioFilename, startTime, endTime, pool, options);
  //computePanning(audioFilename, startTime, endTime, pool, options);
  computeHighlevel(pool, options);
  Pool stats = computeAggregation(pool);
  outputToFile(stats, outputFilename);

}

void computeSegments(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const Pool& options) {

  int lowlevelHopSize = 1024;
  int minimumSegmentsLength = 10;
  int size1 = 1000, inc1 = 300, size2 = 600, inc2 = 50, cpw = 5;

  // compute low level features to feed SBIc
  computeLowLevel(audioFilename, startTime, endTime, pool, options);

  vector<vector<Real> > features;
  try {
    features = pool.value<vector<vector<Real> > >("lowlevel.mfcc");
  }
  catch(const EssentiaException&) {
    cout << "Error: could not find MFCC features in low level pool. Aborting..." << endl;
    exit(3);
  }

  TNT::Array2D<Real> featuresArray(features[0].size(), features.size());
  for (int frame = 0; frame < int(features.size()); ++frame) {
    for (int mfcc = 0; mfcc < int(features[0].size()); ++mfcc) {
      featuresArray[mfcc][frame] = features[frame][mfcc];
    }
  }
  // only BIC segmentation available
  standard::Algorithm* sbic = standard::AlgorithmFactory::create("SBic", "size1", size1, "inc1", inc1,
                                                                 "size2", size2, "inc2", inc2, "cpw", cpw,
                                                                 "minLength", minimumSegmentsLength);
  vector<Real> segments;
  sbic->input("features").set(featuresArray);
  sbic->output("segmentation").set(segments);
  sbic->compute();

  Real analysisSampleRate = options.value<Real>("analysisSampleRate");

  for (int i=0; i<int(segments.size()); ++i) {
    segments[i] *= Real(lowlevelHopSize)/analysisSampleRate;
    pool.add("segmentation.timestamps", segments[i]);
  }
}

void computeReplayGain(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const Pool& options) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Real analysisSampleRate = options.value<Real>("analysisSampleRate");

  /*************************************************************************
   *    1st pass: get metadata and replay gain                             *
   *************************************************************************/

  readMetadata(audioFilename, pool);

  string downmix = "mix";
  Real replayGain = 0.0;
  bool tryReallyHard = true;
  int length = 0;

  while (tryReallyHard) {
    Algorithm* audio_1 = factory.create("EqloudLoader",
                                        "filename", audioFilename,
                                        "sampleRate", analysisSampleRate,
                                        "startTime", startTime,
                                        "endTime", endTime,
                                        "downmix", downmix);

    Algorithm* rgain = factory.create("ReplayGain",
                                      "applyEqloud", false);

    pool.set("metadata.audio_properties.analysis_sample_rate", audio_1->parameter("sampleRate").toReal());
    pool.set("metadata.audio_properties.downmix", downmix);

    connect(audio_1->output("audio"), rgain->input("signal"));
    connect(rgain->output("replayGain"), pool, "metadata.audio_properties.replay_gain");

    cout << "Process step 1: Replay Gain" << endl;
    try {
      Network network(audio_1);
      network.run();
      length = audio_1->output("audio").totalProduced();
      tryReallyHard = false;
    }
    catch (const EssentiaException&) {
      if (downmix == "mix") {
        downmix = "left";
        try {
          pool.remove("metadata.audio_properties.downmix");
          pool.remove("metadata.audio_properties.replay_gain");
        }
        catch (EssentiaException&) {}

        continue;
      }
      else {
        cout << "ERROR: File looks like a completely silent file... Aborting..." << endl;
        exit(4);
      }
    }

    replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");

    // very high value for replayGain, we are probably analyzing a silence even
    // though it is not a pure digital silence
    if (replayGain > 20.0) {
      // NB: except if it was some electro music where someone thought it was smart
      //     to have opposite left and right channels... Try with only the left
      //     channel, then.
      if (downmix == "mix") {
        downmix = "left";
        tryReallyHard = true;
        pool.remove("metadata.audio_properties.downmix");
        pool.remove("metadata.audio_properties.replay_gain");
      }
      else {
        cout << "ERROR: File looks like a completely silent file... Aborting..." << endl;
        exit(5);
      }
    }
  }
  // set length (actually duration) of the file:
  pool.set("metadata.audio_properties.length", length/analysisSampleRate);

  cout.precision(10);
}

void computeLowLevel(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const Pool& options, const string& nspace) {


  /*************************************************************************
   *    2nd pass: normalize the audio with replay gain, compute as         *
   *              many lowlevel descriptors as possible                    *
   *************************************************************************/

  // namespace:
  string llspace = "lowlevel.";
  string rhythmspace = "rhythm.";
  if (!nspace.empty()) {
    llspace = nspace + ".lowlevel.";
    rhythmspace = nspace + ".rhythm.";
  }

  Real replayGain = 0;
  Real analysisSampleRate = options.value<Real>("analysisSampleRate");

  const vector<string>& desc = pool.descriptorNames();
  if (find(desc.begin(), desc.end(), "replayGain") != desc.end()) {
    replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");
  }

  string downmix = "mix";
  try {
    downmix = pool.value<string>("metadata.audio_properties.downmix");
  }
  catch (const EssentiaException&) {
    throw EssentiaException("StreamingExtractor::computeLowLevel, could not determine downmix type");
  }

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_2 = factory.create("EqloudLoader",
                                      "filename", audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime", startTime,
                                      "endTime", endTime,
                                      "replayGain", replayGain,
                                      "downmix", downmix);


  SourceBase& eqloudSource = audio_2->output("audio");

  // Low-Level Spectral Descriptors
  LowLevelSpectral(eqloudSource, pool, options, nspace);

  // Low-Level Spectral Equal Loudness Descriptors
  // expects the audio source to already be equal-loudness filtered
  LowLevelSpectralEqLoud(eqloudSource, pool, options, nspace);

  // Level Descriptor
  // expects the audio source to already be equal-loudness filtered
  Level(eqloudSource, pool, options, nspace);

  // Tuning Frequency
  TuningFrequency(eqloudSource, pool, options, nspace);

  // Rhythm descriptors
  Algorithm* rhythmExtractor = factory.create("RhythmExtractor"); // TODO switch to RhythmExtractor2013?
  connect(eqloudSource, rhythmExtractor->input("signal"));
  connect(rhythmExtractor->output("ticks"), pool, rhythmspace + "beats_position");
  connect(rhythmExtractor->output("bpm"), pool, rhythmspace + "bpm");
  connect(rhythmExtractor->output("estimates"), pool, rhythmspace + "bpm_estimates");
  //connect(rhythmExtractor->output("rubatoStart"), pool, rhythmspace + "rubato_start");
  //connect(rhythmExtractor->output("rubatoStop"), pool, rhythmspace + "rubato_stop");
  //connect(rhythmExtractor->output("rubatoNumber"), pool, rhythmspace + "rubato_sections_number");
  connect(rhythmExtractor->output("bpmIntervals"), pool, rhythmspace + "bpm_intervals");

  // BPM Histogram descriptors
  Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
  connect(rhythmExtractor->output("bpmIntervals"), bpmhist->input("bpmIntervals"));
  connect(bpmhist->output("firstPeakBPM"), pool, rhythmspace + "first_peak_bpm");
  connect(bpmhist->output("firstPeakWeight"), pool, rhythmspace + "first_peak_weight");
  connect(bpmhist->output("firstPeakSpread"), pool, rhythmspace + "first_peak_spread");
  connect(bpmhist->output("secondPeakBPM"), pool, rhythmspace + "second_peak_bpm");
  connect(bpmhist->output("secondPeakWeight"), pool, rhythmspace + "second_peak_weight");
  connect(bpmhist->output("secondPeakSpread"), pool, rhythmspace + "second_peak_spread");

  // Onset Detection
  Algorithm* onset = factory.create("OnsetRate");
  connect(eqloudSource, onset->input("signal"));
  connect(onset->output("onsetTimes"), pool, rhythmspace + "onset_times");
  connect(onset->output("onsetRate"), NOWHERE ); //pool, rhythmspace + "onset_rate"); // this is done later

  cout << "Process step 2: Low Level" << endl;
  Network network(audio_2);
  network.run();


  // check if we processed enough audio for it to be useful, in particular did
  // we manage to get an estimation for the loudness (2 seconds required)
  // We removed this check because we wanted to record at least the rest of the descriptors
  // for short files (i.e. sound effects)
//  try {
//    pool.value<vector<Real> >(llspace + "loudness")[0];
//  }
//  catch (EssentiaException&) {
//    cout << "ERROR: File is too short (< 2sec)... Aborting..." << endl;
//    exit(6);
//  }

  // compute onset rate = len(onsets) / len(audio)
  pool.set(rhythmspace + "onset_rate",
           pool.value<vector<Real> >(rhythmspace + "onset_times").size()
           / (Real)audio_2->output("audio").totalProduced()
           * analysisSampleRate);

}

void computeMidLevel(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const Pool& options, const string& nspace) {

  /*************************************************************************
   *    3rd pass: HPCP & beats loudness (depend on some descriptors that   *
   *              have been computed during the 2nd pass)                  *
   *************************************************************************/
  Real replayGain = 0;
  Real analysisSampleRate = options.value<Real>("analysisSampleRate");

  const vector<string>& desc = pool.descriptorNames();
  if (find(desc.begin(), desc.end(), "replayGain") != desc.end()) {
    replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");
  }
  string downmix = "mix";
  try {
    downmix = pool.value<string>("metadata.audio_properties.downmix");
  }
  catch (const EssentiaException&) {
    throw EssentiaException("StreamingExtractor::computeLowLevel, could not determine downmix type");
  }

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_3 = factory.create("EqloudLoader",
                                               "filename", audioFilename,
                                               "sampleRate", analysisSampleRate,
                                               "startTime", startTime,
                                               "endTime", endTime,
                                               "replayGain", replayGain,
                                               "downmix", downmix);


  SourceBase& audioSource_2 = audio_3->output("audio");

  // Compute Tonal descriptors (needed TuningFrequency before)
  TonalDescriptors(audioSource_2, pool, options, nspace);


  // Compute the loudness at the beats position (needed beats position)
  string rhythmspace = "rhythm.";
  if (!nspace.empty()) rhythmspace = nspace + ".rhythm.";
  vector<Real> ticks = pool.value<vector<Real> >(rhythmspace + "beats_position");

  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                                     "sampleRate", analysisSampleRate,
                                                     "beats", ticks);

  connect(audioSource_2, beatsLoudness->input("signal"));
  connect(beatsLoudness->output("loudness"), pool, rhythmspace + "beats_loudness");
  connect(beatsLoudness->output("loudnessBandRatio"), pool, rhythmspace + "beats_loudness_band_ratio");

  cout << "Process step 3: Mid Level" << endl;
  Network network(audio_3);
  network.run();
}

void computePanning(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const Pool& options, const string& nspace) {

  /*************************************************************************
   *    4th pass: Panning                                                  *
   *                                                                       *
   *************************************************************************/

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_4 = factory.create("AudioLoader",
                                               "filename", audioFilename);
  SourceBase& audioSource_3 = audio_4->output("audio");
  connect(audio_4->output("sampleRate"), NOWHERE);
  connect(audio_4->output("numberChannels"), NOWHERE);
  Panning(audioSource_3, pool, options, nspace);
  cout << "Process step 4: Panning" << endl;
  Network network(audio_4);
  network.run();
}

void computeHighlevel(Pool& pool, const Pool& options, const string& nspace) {

  /*************************************************************************
   *    5th pass: High-level descriptors that depend on others, but we     *
   *              don't need to stream the audio anymore                   *
   *************************************************************************/

  cout << "Process step 4: HighLevel" << endl;

  // Average Level
  LevelAverage(pool, nspace);

  // SFX Descriptors
  SFXPitch(pool, nspace);

  // Tuning System Features
  TuningSystemFeatures(pool, nspace);

  // Pool Cleaning (remove temporary descriptors)
  TonalPoolCleaning(pool, nspace);

  // Add missing descriptors which are not computed yet, but will be for the
  // final release or during the 1.x cycle. However, the schema need to be
  // complete before that, so just put default values for these.
  PostProcess(pool, options, nspace);
}

Pool computeAggregation(Pool& pool, int nSegments) {

  // choose which descriptors stats to output
  const char* stats[] = { "mean", "var", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };
  const char* mfccStats[] = { "mean", "cov", "icov" };
  vector<string> value(1);
  value[0] = "copy";

  map<string, vector<string> > exceptions;
  exceptions["lowlevel.mfcc"] = arrayToVector<string>(mfccStats);

  // in case there is segmentation:

  if (nSegments > 0) {
    exceptions["segmentation.timestamps"] = value;
    for (int i=0; i<nSegments; i++) {
      ostringstream ns;
      ns << "segment_" << i;
      exceptions[ns.str() + ".lowlevel.mfcc"] = value;
    }
  }

  standard::Algorithm* aggregator = standard::AlgorithmFactory::create("PoolAggregator",
                                                             "defaultStats", arrayToVector<string>(stats),
                                                             "exceptions", exceptions);

  Pool poolStats;

  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);

  cout << "Process step 5: Aggregation" << endl;

  aggregator->compute();

  delete aggregator;

  return poolStats;
}

void outputToFile(Pool& pool, const string& outputFilename) {

  cout << "Writing results to file " << outputFilename << endl;
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                         "filename", outputFilename);
  output->input("pool").set(pool);
  output->compute();

  delete output;
}
