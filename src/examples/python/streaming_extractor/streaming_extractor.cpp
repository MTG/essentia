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
#include <algorithm> // for find()
// essentia
#include "algorithmfactory.h"
#include "poolstorage.h"

// helper functions
#include "streaming_extractorutils.h"

// composites translated from python:
#include "streaming_extractorlowlevelspectral.h"
#include "streaming_extractorlowlevelspectraleqloud.h"
#include "streaming_extractorlevel.h"
#include "streaming_extractortonaldescriptors.h"
#include "streaming_extractorpitch.h"
#include "streaming_extractortuningfrequency.h"
#include "streaming_extractorrhythmdescriptors.h"
#include "streaming_extractorsfx.h"
#include "streaming_extractorpanning.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

//typedef essentia::standard::IOMap<essentia::streaming::SourceBase*> OutputMap;
typedef EssentiaMap<std::string, SourceBase*, string_cmp> OutputMap;


// global variables:
int _lowlevelFrameSize = 2048;
int _lowlevelHopSize = 1024;
int _tonalFrameSize = 4096;
int _tonalHopSize = 2048;
int _dynamicsFrameSize = 88200;
int _dynamicsHopSize = 44100;
int _panningFrameSize = 8192;
int _panningHopSize = 2048;

void computeSegments(const string& audioFilename, Real startTime, Real endTime, Pool& pool);
void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& pool);

void computeReplayGain(const string& audioFilename, Real startTime, Real endTime, Pool& pool);
void computeLowLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& pool, const string& nspace = "");
void computeMidLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& pool, const string& nspace = "");
void computePanning(const string& audioFilename, Real startTime, Real endTime,
                    Pool& pool, const string& nspace = "");
void computeHighlevel(Pool& pool, const string& nspace = "");
Pool computeAggregation(Pool& pool, int segments=0);
void addSVMDescriptors(Pool& pool);
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
        cout << "Unrecognized option \"" << argv[3] << "\"" << endl;
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

  // pool for storing results
  Pool pool;
  // this pool contains only descriptors computed after applying equal loudness
  // filetering:
  pool.set("metadata.audio_properties.equal_loudness", true);

  if (computeSegmentation) {
    // pool for storing segments:
    computeReplayGain(audioFilename, startTime, endTime, pool);
    computeSegments(audioFilename, startTime, endTime, pool);
    vector<Real> segments = pool.value<vector<Real> >("segmentation.timestamps");
    for (int i=0; i<int(segments.size()-1); ++i) {
      Real start = segments[i];
      Real end = segments[i+1];
      cout << "\n**************************************************************************";
      cout << "\nSegment " << i << ": processing audio from " << start << "s to " << end << "s";
      cout << "\n**************************************************************************" << endl;

      // set segment name
      ostringstream ns;
      ns << "segment_" << i;
      string sn = ns.str();
      ns.str(""); ns << "segments." << sn << ".name";
      pool.set(ns.str(), sn);

      // set segment scope
      ns.str(""); ns << "segments." << sn << ".scope";
      vector<Real> scope(2, 0);
      scope[0] = start;
      scope[1] = end;
      pool.set(ns.str(), scope);

      // compute descriptors
      ns.str(""); ns << "segments.segment_" << i << ".descriptors";

      computeLowLevel(audioFilename, start, end, pool, ns.str());
      computeMidLevel(audioFilename, start, end, pool, ns.str());
      //computePanning(audioFilename, start, end, pool, ns.str());
      computeHighlevel(pool, ns.str());
    }
    if (endTime > pool.value<Real>("metadata.audio_properties.length")) {
      endTime = pool.value<Real>("metadata.audio_properties.length");
    }
    cout << "\n**************************************************************************\n";
    //cout << "\n compute the rest of descriptors for the entire audio from " << startTime << "s to " << endTime << "s";
    //cout << "\n**************************************************************************" << endl;
    //computeLowLevel(audioFilename, startTime, endTime, pool); // already computed when performing segmentation
    computeMidLevel(audioFilename, startTime, endTime, pool);
    computeHighlevel(pool);

    Pool stats = computeAggregation(pool, segments.size());
    // Add this line when svm models are trained
    //addSVMDescriptors(stats);
    outputToFile(stats, outputFilename);
  }
  else {
    try {
      compute(audioFilename, outputFilename, startTime, endTime, pool);
    }
    catch (EssentiaException& e) {
      cout << e.what() << endl;
      throw;
    }
  }

  pool.remove("metadata.audio_properties.downmix");
  essentia::shutdown();

  return 0;
}

void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& pool) {
  computeReplayGain(audioFilename, startTime, endTime, pool);
  computeLowLevel(audioFilename, startTime, endTime, pool);
  computeMidLevel(audioFilename, startTime, endTime, pool);
  //computePanning(audioFilename, startTime, endTime, pool);
  computeHighlevel(pool);
  Pool stats = computeAggregation(pool);
  // Add this line when svm models are trained
  //addSVMDescriptors(stats);
  outputToFile(stats, outputFilename);

}

void computeSegments(const string& audioFilename, Real startTime, Real endTime, Pool& pool) {

  int lowlevelHopSize = 1024;
  int minimumSegmentsLength = 10;
  int size1 = 1000, inc1 = 300, size2 = 600, inc2 = 50, cpw = 5;

  // compute low level features to feed SBIc
  computeLowLevel(audioFilename, startTime, endTime, pool);

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
  Real analysisSampleRate = 44100;
  try {
    analysisSampleRate = pool.value<Real>("metadata.audio_properties.analysis_sample_rate");
  }
  catch(const EssentiaException&) {
    throw EssentiaException("Warning: StreamingExtractor::computeSegments, could not find analysis sampling rate");
  }
  for (int i=0; i<int(segments.size()); ++i) {
    segments[i] *= Real(lowlevelHopSize)/analysisSampleRate;
    pool.add("segmentation.timestamps", segments[i]);
  }
}

void computeReplayGain(const string& audioFilename, Real startTime, Real endTime, Pool& pool) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Real analysisSampleRate = 44100;


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
      runGenerator(audio_1);
      length = audio_1->output("audio").totalProduced();
      deleteNetwork(audio_1);
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

void computeLowLevel(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const string& nspace) {


  /*************************************************************************
   *    2nd pass: normalize the audio with replay gain, compute as         *
   *              many lowlevel descriptors as possible                    *
   *************************************************************************/

  // namespace:
  string llspace = "lowlevel.";
  string rhythmspace = "rhythm.";
  string sfxspace = "sfx.";
  string tonalspace = "tonal.";
  if (!nspace.empty()) {
    llspace = nspace + ".lowlevel.";
    rhythmspace = nspace + ".rhythm.";
    sfxspace = nspace + ".sfx.";
    tonalspace = nspace + ".tonal.";
  }

  Real replayGain = 0;
  Real analysisSampleRate = 44100;
  string downmix = "mix";
  getAnalysisData(pool, replayGain, analysisSampleRate, downmix);


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
  Algorithm * lowLevelSpectral = new LowLevelSpectralExtractor();
  lowLevelSpectral->configure("frameSize", _lowlevelFrameSize,
                              "hopSize", _lowlevelHopSize,
                              "halfSampleRate", analysisSampleRate*0.5);

  // connect inputs:
  connect(eqloudSource, lowLevelSpectral->input("signal"));
  // connect outputs:
  const char * sfxDescArray[] = {"inharmonicity", "oddtoevenharmonicenergyratio", "tristimulus"};
  vector<string> sfxDesc = arrayToVector<string>(sfxDescArray);
  OutputMap::const_iterator it = lowLevelSpectral->outputs().begin();
  for (; it != lowLevelSpectral->outputs().end(); ++it) {
    string output_name = it->first;
    string ns = llspace; // namespace
    if (find(sfxDesc.begin(), sfxDesc.end(), output_name) != sfxDesc.end()) {
      ns = sfxspace;
    }
    connect(*it->second, pool, ns + output_name);
  }
  // alternatively:
  // connect(lowLevelSpectral->output("silence_rate_20dB"), pool, llspace + "silence_rate_20dB");
  // connect(lowLevelSpectral->output("silence_rate_30dB"), pool, llspace + "silence_rate_30dB");
  // etc.

  // Low-Level Spectral Equal Loudness Descriptors
  Algorithm * lowLevelSpectralEqloud = new LowLevelSpectralEqloudExtractor();
  lowLevelSpectralEqloud->configure("frameSize", _lowlevelFrameSize,
                                    "hopSize", _lowlevelHopSize,
                                    "sampleRate", analysisSampleRate,
                                    "halfSampleRate", analysisSampleRate*0.5);

  // connect inputs:
  connect(eqloudSource, lowLevelSpectralEqloud->input("signal"));
  // connect outputs:
  it = lowLevelSpectralEqloud->outputs().begin();
  for (; it != lowLevelSpectralEqloud->outputs().end(); ++it) {
    connect(*it->second, pool, llspace + it->first);
  }

  // Level Descriptor
  Algorithm * level = new LevelExtractor();
  level->configure("frameSize", _dynamicsFrameSize,
                   "hopSize", _dynamicsHopSize);

  connect(eqloudSource, level->input("signal"));
  connect(level->output("loudness"), pool, llspace + "loudness");

  // Tuning Frequency
  Algorithm * tuningFrequency = new TuningFrequencyExtractor();
  tuningFrequency->configure("frameSize", _tonalFrameSize,
                             "hopSize", _tonalHopSize);

  connect(eqloudSource, tuningFrequency->input("signal"));
  connect(tuningFrequency->output("tuningFrequency"), pool, tonalspace + "tuning_frequency");

  // Rhythm descriptors & BPM Histogram descriptors:
  Algorithm * rhythmExtractor = new RhythmDescriptorsExtractor();
  rhythmExtractor->configure();

  connect(eqloudSource, rhythmExtractor->input("signal"));
  it = rhythmExtractor->outputs().begin();
  for (; it != rhythmExtractor->outputs().end(); ++it) {
    connect(*it->second, pool, rhythmspace + it->first);
  }

  // Onset Detection
  Algorithm* onset = factory.create("OnsetRate");
  connect(eqloudSource, onset->input("signal"));
  connect(onset->output("onsetTimes"), pool, rhythmspace + "onset_times");
  connect(onset->output("onsetRate"), NOWHERE );

  cout << "Process step 2: Low Level" << endl;
  runGenerator(audio_2);

  // check if we processed enough audio for it to be useful, in particular did
  // we manage to get an estimation for the loudness (2 seconds required)
  try {
    pool.value<vector<Real> >(llspace + "loudness")[0];
  }
  catch (const EssentiaException& e) {
    cout << "ERROR: File is too short (< 2sec)... Aborting..." << endl;
    exit(6);
  }

  // compute onset rate = len(onsets) / len(audio)
  pool.set(rhythmspace + "onset_rate", pool.value<vector<Real> >(rhythmspace + "onset_times").size()
     / (Real)audio_2->output("audio").totalProduced()
     * pool.value<Real>("metadata.audio_properties.analysis_sample_rate"));


  // delete network only now, because we needed audio_2->output("audio") to
  // compute the onset rate on the previous line.
  deleteNetwork(audio_2);
}

void computeMidLevel(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const string& nspace) {

  /*************************************************************************
   *    3rd pass: HPCP & beats loudness (depend on some descriptors that   *
   *              have been computed during the 2nd pass)                  *
   *************************************************************************/
  string tonalspace = "tonal.";
  string rhythmspace = "rhythm.";
  if (!nspace.empty()) {
    rhythmspace = nspace + ".rhythm.";
    tonalspace = nspace + ".tonal.";
  }
  Real replayGain = 0;
  Real analysisSampleRate = 44100;
  string downmix = "mix";
  getAnalysisData(pool, replayGain, analysisSampleRate, downmix);

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
  Real tuningFreq = pool.value<vector<Real> >(tonalspace + "tuning_frequency").back();
  Algorithm * tonalDescriptors = new TonalDescriptorsExtractor();
  tonalDescriptors->configure("frameSize", _tonalFrameSize,
                              "hopSize", _tonalHopSize,
                              "tuningFrequency", tuningFreq);

  connect(audioSource_2, tonalDescriptors->input("signal"));
  OutputMap::const_iterator it = tonalDescriptors->outputs().begin();
  for (; it != tonalDescriptors->outputs().end(); ++it) {
    connect(*it->second, pool, tonalspace + it->first);
  }

  // Compute the loudness at the beats position (needed beats position)
  vector<Real> ticks = pool.value<vector<Real> >(rhythmspace + "beats_position");

  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                            "sampleRate", analysisSampleRate,
                                            "beats", ticks);

  connect(audioSource_2, beatsLoudness->input("signal"));
  connect(beatsLoudness->output("loudness"), pool, rhythmspace + "beats_loudness");
  connect(beatsLoudness->output("loudnessBass"), pool, rhythmspace + "beats_loudness_bass");

  cout << "Process step 3: Mid Level" << endl;
  runGenerator(audio_3);
  deleteNetwork(audio_3);
}

void computePanning(const string& audioFilename, Real startTime, Real endTime, Pool& pool, const string& nspace) {

  /*************************************************************************
   *    4th pass: Panning                                                  *
   *                                                                       *
   *************************************************************************/
  Real analysisSampleRate = 44100.0;
  try {
    analysisSampleRate = pool.value<Real>("metadata.audio_properties.analysis_sample_rate");
  }
  catch(const EssentiaException&) {
    throw EssentiaException("Warning: StreamingExtractor::computePanning, could not find analysis sampling rate");
  }

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_4 = factory.create("AudioLoader",
                                      "filename", audioFilename);
  Algorithm* trimmer = factory.create("Trimmer",
                                      "startTime", startTime,
                                      "endTime", endTime);
  connect(audio_4->output("audio"), trimmer->input("signal"));
  connect(audio_4->output("sampleRate"), NOWHERE);
  connect(audio_4->output("numberChannels"), NOWHERE);
  Algorithm* panningExtractor = factory.create("PanningExtractor",
                                               "frameSize", _panningFrameSize,
                                               "hopSize", _panningHopSize,
                                               "sampleRate", analysisSampleRate);

  connect(trimmer->output("signal"), panningExtractor->input("signal"));
  connect(panningExtractor->output("panning_coefficients"), pool, "panning_coefficients");

  runGenerator(audio_4);
  deleteNetwork(audio_4);
}

void computeHighlevel(Pool& pool, const string& nspace) {

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
  PostProcess(pool, nspace);
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
      ns << "segments.segment_" << i << ".descriptors";
      exceptions[ns.str() + ".lowlevel.mfcc"] = arrayToVector<string>(mfccStats);
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
                                                                   "filename", outputFilename,
                                                                   "doubleCheck", true);
  output->input("pool").set(pool);
  output->compute();

  delete output;
}


//void addSVMDescriptors(Pool& pool) {
//  //const char* svmModels[] = {}; // leave this empty if you don't have any SVM models
//  const char* svmModels[] = { "BAL", "CUL", "GDO", "GRO", "GTZ", "PS", "VI",
//                              "MAC", "MAG", "MEL", "MHA", "MPA", "MRE", "MSA" };
//
//  string pathToSvmModels;
//
//#ifdef OS_WIN32
//  pathToSvmModels = "svm_models\\";
//#else
//  pathToSvmModels = "svm_models/";
//#endif
//
//  for (int i=0; i<(int)ARRAY_SIZE(svmModels); i++) {
//    string modelFilename = pathToSvmModels + string(svmModels[i]) + ".model";
//    standard::Algorithm* svm = standard::AlgorithmFactory::create("SvmClassifier",
//                                                                  "model", modelFilename);
//
//    svm->input("pool").set(pool);
//    svm->output("result").set(pool);
//    svm->compute();
//
//    delete svm;
//  }
//}
