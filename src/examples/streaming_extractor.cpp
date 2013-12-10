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

#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiautil.h>
#include <essentia/scheduler/network.h>

// helper functions
#include "streaming_extractorutils.h"
#include "streaming_extractorlowlevel.h"
//#include "streaming_extractorbeattrack.h" // outdated beat tracker
#include "streaming_extractorsfx.h"
#include "streaming_extractortonal.h"
#include "streaming_extractorpanning.h"
#include "streaming_extractorpostprocess.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

void computeSegments(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options);
void compute(const string& audioFilename, const string& outputFilename,
             Pool& neqloudPool, Pool& eqloudPool, const Pool& options);

void computeReplayGain(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options);
void computeLowLevel(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options,
                     Real startTime, Real endTime, const string& nspace = "");
void computeBeatTrack(Pool& pool, const Pool& options, const string& nspace = "");
void computeMidLevel(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options,
                     Real startTime, Real endTime, const string& nspace = "");
void computePanning(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options,
                    Real startTime, Real endTime, const string& nspace = "");
void computeHighlevel(Pool& pool, const Pool& options, const string& nspace = "");
Pool computeAggregation(Pool& pool, const Pool& options, int segments=0);
void addSVMDescriptors(Pool& pool);
void outputToFile(Pool& pool, const string& outputFilename, const Pool& options);

void usage() {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: streaming_extractor input_audiofile output_textfile [profile]" << endl;
    exit(1);
}


int main(int argc, char* argv[]) {

  string audioFilename, outputFilename, profileFilename;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    case 4: // profile supplied
      audioFilename =  argv[1];
      outputFilename = argv[2];
      profileFilename = argv[3];
      break;
    default:
      usage();
  }

  // Register the algorithms in the factory(ies)
  essentia::init();

  //setDebugLevel(EExecution);
  /*
  setDebugLevel(EAll);
  unsetDebugLevel(EExecution | EMemory);
  unsetDebugLevel(ENetwork);
  unsetDebugLevel(EConnectors);
  */

  // pool cotaining profile (configure) options:
  Pool options;

  // set configuration from file or otherwise use default settings:
  setOptions(options, profileFilename);

  // pool for storing results
  Pool neqloudPool; // non equal loudness pool
  Pool eqloudPool; // equal loudness pool

  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud =  options.value<Real>("equalLoudness")  != 0;

  if (!eqloud && !neqloud) {
    throw EssentiaException("Configuration for both equal loudness and non\
       equal loudness is set to false. At least one must set to true");
  }

  try {
    compute(audioFilename, outputFilename, neqloudPool, eqloudPool, options);
  }
  catch (EssentiaException& e) {
    cout << e.what() << endl;
    throw;
  }

  essentia::shutdown();

  return 0;
}

void compute(const string& audioFilename, const string& outputFilename,
             Pool& neqloudPool, Pool& eqloudPool, const Pool& options) {

  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud = options.value<Real>("equalLoudness") != 0;

  if (neqloud) neqloudPool.set("metadata.audio_properties.equal_loudness", false);
  if (eqloud) eqloudPool.set("metadata.audio_properties.equal_loudness", true);

  // what to compute:
  bool lowlevel = options.value<Real>("lowlevel.compute")         ||
                  options.value<Real>("average_loudness.compute") ||
                  options.value<Real>("tonal.compute")            ||
                  options.value<Real>("sfx.compute")              ||
                  options.value<Real>("rhythm.compute");
  //bool beattrack = options.value<Real>("beattrack.compute");
  bool midlevel = options.value<Real>("tonal.compute") ||
                  options.value<Real>("rhythm.compute");
  bool panning = options.value<Real>("panning.compute") != 0;

  // compute features for the whole song
  computeReplayGain(audioFilename, neqloudPool, eqloudPool, options);
  Real startTime = options.value<Real>("startTime");
  Real endTime = options.value<Real>("endTime");
  if (eqloud) {
    if (endTime > eqloudPool.value<Real>("metadata.audio_properties.length")) {
      endTime = eqloudPool.value<Real>("metadata.audio_properties.length");
    }
  }
  else {
    if (endTime > neqloudPool.value<Real>("metadata.audio_properties.length")) {
      endTime = neqloudPool.value<Real>("metadata.audio_properties.length");
    }
  }
  if (lowlevel)
    computeLowLevel(audioFilename, neqloudPool, eqloudPool, options, startTime, endTime);
  // outdated beat tracker
  //if (beattrack) {
  //  if (neqloud) computeBeatTrack(neqloudPool, options);
  //  if (eqloud) computeBeatTrack(eqloudPool, options);
  //}
  if (midlevel)
    computeMidLevel(audioFilename, neqloudPool, eqloudPool, options, startTime, endTime);
  if (panning)
    computePanning(audioFilename, neqloudPool, eqloudPool, options, startTime, endTime);
  if (neqloud) computeHighlevel(neqloudPool, options);
  if (eqloud) computeHighlevel(eqloudPool, options);

  vector<Real> segments;
  if (options.value<Real>("segmentation.compute") != 0) {
    computeSegments(audioFilename, neqloudPool, eqloudPool, options);
    segments = eqloudPool.value<vector<Real> >("segmentation.timestamps");
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
      if (neqloud) neqloudPool.set(ns.str(), sn);
      if (eqloud) eqloudPool.set(ns.str(), sn);

      // set segment scope
      ns.str(""); ns << "segments." << sn << ".scope";
      vector<Real> scope(2, 0);
      scope[0] = start;
      scope[1] = end;
      if (neqloud) neqloudPool.set(ns.str(), scope);
      if (eqloud) eqloudPool.set(ns.str(), scope);

      // compute descriptors
      ns.str(""); ns << "segments.segment_" << i << ".descriptors";

      if (lowlevel)
        computeLowLevel(audioFilename, neqloudPool, eqloudPool, options, start, end, ns.str());
      // outdated beat tracker
      //if (beattrack) {
      //  if (neqloud) computeBeatTrack(neqloudPool, options);
      //  if (eqloud) computeBeatTrack(eqloudPool, options);
      //}
      if (midlevel)
        computeMidLevel(audioFilename, neqloudPool, eqloudPool, options, start, end, ns.str());
      if (panning)
        computePanning(audioFilename, neqloudPool, eqloudPool, options, start, end, ns.str());
      if (neqloud) computeHighlevel(neqloudPool, options);
      if (eqloud) computeHighlevel(eqloudPool, options);
    }
    cout << "\n**************************************************************************\n";
  }

 if (neqloud) {
   string baseFilename = outputFilename.substr(0, outputFilename.size()-4);
   string neqOutputFilename = baseFilename + ".neq.sig";
   Pool stats = computeAggregation(neqloudPool, options, segments.size());
   //if (options.value<Real>("svm.compute") != 0) addSVMDescriptors(stats); //not available
   outputToFile(stats, neqOutputFilename, options);
   neqloudPool.remove("metadata.audio_properties.downmix");
 }

 if (eqloud) {
   Pool stats = computeAggregation(eqloudPool, options, segments.size());
   if (options.value<Real>("svm.compute") != 0) addSVMDescriptors(stats);
   outputToFile(stats, outputFilename, options);
   eqloudPool.remove("metadata.audio_properties.downmix");
 }
 return;
}

void computeSegments(const string& audioFilename, Pool& neqloudPool,
                     Pool& eqloudPool, const Pool& options) {

  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud =  options.value<Real>("equalLoudness")  != 0;

  int minimumSegmentsLength = int(options.value<Real>("segmentation.minimumSegmentsLength"));
  int size1 = int(options.value<Real>("segmentation.size1"));
  int inc1  = int(options.value<Real>("segmentation.inc1"));
  int size2 = int(options.value<Real>("segmentation.size2"));
  int inc2  = int(options.value<Real>("segmentation.inc2"));
  int cpw   = int(options.value<Real>("segmentation.cpw"));

  vector<vector<Real> > features;
  try {
    if (eqloud)
      features = eqloudPool.value<vector<vector<Real> > >("lowlevel.mfcc");
    else features = neqloudPool.value<vector<vector<Real> > >("lowlevel.mfcc");
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
  Real analysisSampleRate = options.value<Real>("sampleRate");
  Real step = options.value<Real>("lowlevel.hopSize");
  for (int i=0; i<int(segments.size()); ++i) {
    segments[i] *= step/analysisSampleRate;
    if (neqloud) neqloudPool.add("segmentation.timestamps", segments[i]);
    if (eqloud) eqloudPool.add("segmentation.timestamps", segments[i]);
  }
}

void computeReplayGain(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool, const Pool& options) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Real analysisSampleRate = options.value<Real>("analysisSampleRate");


  /*************************************************************************
   *    1st pass: get metadata and replay gain                             *
   *************************************************************************/
  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud =  options.value<Real>("equalLoudness")  != 0;

  if (neqloud) readMetadata(audioFilename, neqloudPool);
  if (eqloud) readMetadata(audioFilename, eqloudPool);
  Real startTime = options.value<Real>("startTime");
  Real endTime = options.value<Real>("endTime");

  string downmix = "mix";
  Real replayGain = 0.0;
  bool tryReallyHard = true;
  int length = 0;

  while (tryReallyHard) {
    Algorithm* audio_1 = factory.create("EqloudLoader",
                                        "filename",   audioFilename,
                                        "sampleRate", analysisSampleRate,
                                        "startTime",  startTime,
                                        "endTime",    endTime,
                                        "downmix",    downmix);

    Algorithm* rgain   = factory.create("ReplayGain",
                                        "applyEqloud", false);

    if (neqloud) {
      neqloudPool.set("metadata.audio_properties.analysis_sample_rate", audio_1->parameter("sampleRate").toReal());
      neqloudPool.set("metadata.audio_properties.downmix", downmix);
    }
    if (eqloud) {
      eqloudPool.set("metadata.audio_properties.analysis_sample_rate", audio_1->parameter("sampleRate").toReal());
      eqloudPool.set("metadata.audio_properties.downmix", downmix);
    }

    audio_1->output("audio")  >>  rgain->input("signal");
    if (neqloud)
      rgain->output("replayGain")  >>  PC(neqloudPool, "metadata.audio_properties.replay_gain");
    if (eqloud)
      rgain->output("replayGain")  >>  PC(eqloudPool, "metadata.audio_properties.replay_gain");

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
          neqloudPool.remove("metadata.audio_properties.downmix");
          neqloudPool.remove("metadata.audio_properties.replay_gain");
          eqloudPool.remove("metadata.audio_properties.downmix");
          eqloudPool.remove("metadata.audio_properties.replay_gain");
        }
        catch (EssentiaException&) {}

        continue;
      }
      else {
        cout << "ERROR: File looks like a completely silent file... Aborting..." << endl;
        exit(4);
      }
    }

    if (eqloud) replayGain = eqloudPool.value<Real>("metadata.audio_properties.replay_gain");
    else replayGain = neqloudPool.value<Real>("metadata.audio_properties.replay_gain");

    // very high value for replayGain, we are probably analyzing a silence even
    // though it is not a pure digital silence
    if (replayGain > 40.0) { // before it was set to 20 but it was found too conservative
      // NB: except if it was some electro music where someone thought it was smart
      //     to have opposite left and right channels... Try with only the left
      //     channel, then.
      if (downmix == "mix") {
        downmix = "left";
        tryReallyHard = true;
        neqloudPool.remove("metadata.audio_properties.downmix");
        neqloudPool.remove("metadata.audio_properties.replay_gain");
        eqloudPool.remove("metadata.audio_properties.downmix");
        eqloudPool.remove("metadata.audio_properties.replay_gain");
      }
      else {
        cout << "ERROR: File looks like a completely silent file... Aborting..." << endl;
        exit(5);
      }
    }
  }
  // set length (actually duration) of the file:
  if (neqloud) neqloudPool.set("metadata.audio_properties.length", length/analysisSampleRate);
  if (eqloud) eqloudPool.set("metadata.audio_properties.length", length/analysisSampleRate);

  cout.precision(10);
}

void computeLowLevel(const string& audioFilename, Pool& neqloudPool, Pool& eqloudPool,
                     const Pool& options, Real startTime, Real endTime, const string& nspace) {
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
  string downmix = "mix";

  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud =  options.value<Real>("equalLoudness")  != 0;
  bool shortsound = options.value<Real>("shortSound")  != 0;

  eqloud = true;
  neqloud = false;
  shortsound = true;

  if (eqloud) {
    replayGain = eqloudPool.value<Real>("metadata.audio_properties.replay_gain");
    downmix = eqloudPool.value<string>("metadata.audio_properties.downmix");
  }
  else {
    replayGain = neqloudPool.value<Real>("metadata.audio_properties.replay_gain");
    downmix = neqloudPool.value<string>("metadata.audio_properties.downmix");
  }

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_2 = factory.create("EasyLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime",  startTime,
                                      "endTime",    endTime,
                                      "replayGain", replayGain,
                                      "downmix",    downmix);

  SourceBase& neqloudSource = audio_2->output("audio");
  Algorithm* eqloud2 = factory.create("EqualLoudness");
  connect(audio_2->output("audio"), eqloud2->input("signal"));
  SourceBase& eqloudSource = eqloud2->output("signal");

  if (neqloud) {
    LowLevelSpectral(neqloudSource, neqloudPool, options, nspace);

    // Low-Level Spectral Equal Loudness Descriptors
    // expects the audio source to already be equal-loudness filtered, so it
    // must use the eqloudSouce instead of neqloudSource
    LowLevelSpectralEqLoud(eqloudSource, neqloudPool, options, nspace);

    // Level Descriptor
    // expects the audio source to already be equal-loudness filtered, so it
    // must use the eqloudSouce instead of neqloudSource
    if (options.value<Real>("average_loudness.compute") != 0)
      Level(eqloudSource, neqloudPool, options, nspace);

    // Tuning Frequency
    if (options.value<Real>("tonal.compute") != 0)
      TuningFrequency(neqloudSource, neqloudPool, options, nspace);

    // Rhythm descriptors
    if (options.value<Real>("rhythm.compute") != 0) {
      Algorithm* rhythmExtractor = factory.create("RhythmExtractor2013");
      rhythmExtractor->configure("method", options.value<string>("rhythm.method"),
                                 "maxTempo", options.value<Real>("rhythm.maxTempo"),
                                 "minTempo", options.value<Real>("rhythm.minTempo"));

      // Outdated rhythm extraction algorithm
      //Algorithm* rhythmExtractor = factory.create("RhythmExtractor");
      //rhythmExtractor->configure("useOnset",     options.value<Real>("rhythm.useOnset") != 0,
      //                           "useBands",     options.value<Real>("rhythm.useBands") != 0,
      //                           "frameSize",     int(options.value<Real>("rhythm.frameSize")),
      //                           "hopSize",       int(options.value<Real>("rhythm.hopSize")),
      //                           "numberFrames",  int(options.value<Real>("rhythm.numberFrames")),
      //                           "frameHop",      int(options.value<Real>("rhythm.frameHop")));

      connect(neqloudSource, rhythmExtractor->input("signal"));
      connect(rhythmExtractor->output("ticks"),        neqloudPool, rhythmspace + "beats_position");
      connect(rhythmExtractor->output("bpm"),          neqloudPool, rhythmspace + "bpm");
      connect(rhythmExtractor->output("estimates"),    neqloudPool, rhythmspace + "bpm_estimates");
      // TODO we need a better rubato estimation algorithm
      //connect(rhythmExtractor->output("rubatoStart"),  neqloudPool, rhythmspace + "rubato_start");
      //connect(rhythmExtractor->output("rubatoStop"),   neqloudPool, rhythmspace + "rubato_stop");
      connect(rhythmExtractor->output("bpmIntervals"), neqloudPool, rhythmspace + "bpm_intervals");
      // discard dummy value for confidence as 'degara' beat tracker is not 
      // able to compute it
      rhythmExtractor->output("confidence") >> NOWHERE; 


      // BPM Histogram descriptors
      Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
      connect(rhythmExtractor->output("bpmIntervals"), bpmhist->input("bpmIntervals"));
      connectSingleValue(bpmhist->output("firstPeakBPM"),     neqloudPool, rhythmspace + "first_peak_bpm");
      connectSingleValue(bpmhist->output("firstPeakWeight"),  neqloudPool, rhythmspace + "first_peak_weight");
      connectSingleValue(bpmhist->output("firstPeakSpread"),  neqloudPool, rhythmspace + "first_peak_spread");
      connectSingleValue(bpmhist->output("secondPeakBPM"),    neqloudPool, rhythmspace + "second_peak_bpm");
      connectSingleValue(bpmhist->output("secondPeakWeight"), neqloudPool, rhythmspace + "second_peak_weight");
      connectSingleValue(bpmhist->output("secondPeakSpread"), neqloudPool, rhythmspace + "second_peak_spread");

      // Onset Detection
      Algorithm* onset = factory.create("OnsetRate");
      connect(neqloudSource, onset->input("signal"));
      connect(onset->output("onsetTimes"), neqloudPool, rhythmspace + "onset_times");
      connect(onset->output("onsetRate"), NOWHERE ); //pool, rhythmspace + "onset_rate"); // this is done later
    }
  }

  if (eqloud) {

    // Low-Level Spectral Descriptors
    LowLevelSpectral(eqloudSource, eqloudPool, options, nspace);

    // Low-Level Spectral Equal Loudness Descriptors
    // expects the audio source to already be equal-loudness filtered
    LowLevelSpectralEqLoud(eqloudSource, eqloudPool, options, nspace);

    // Level Descriptor
    // expects the audio source to already be equal-loudness filtered
    if (options.value<Real>("average_loudness.compute") != 0)
      Level(eqloudSource, eqloudPool, options, nspace);

    // Tuning Frequency
    if (options.value<Real>("tonal.compute") != 0)
      TuningFrequency(eqloudSource, eqloudPool, options, nspace);

    // Rhythm descriptors
    if (options.value<Real>("rhythm.compute") != 0) {
      Algorithm* rhythmExtractor = factory.create("RhythmExtractor2013");
      rhythmExtractor->configure("method", options.value<string>("rhythm.method"),
                                 "maxTempo", options.value<Real>("rhythm.maxTempo"),
                                 "minTempo", options.value<Real>("rhythm.minTempo"));

      // Outdated rhythm extraction algorithm
      //Algorithm* rhythmExtractor = factory.create("RhythmExtractor");
      //rhythmExtractor->configure("useOnset",     options.value<Real>("rhythm.useOnset") != 0,
      //                           "useBands",     options.value<Real>("rhythm.useBands") != 0,
      //                           "frameSize",     int(options.value<Real>("rhythm.frameSize")),
      //                           "hopSize",       int(options.value<Real>("rhythm.hopSize")),
      //                           "numberFrames",  int(options.value<Real>("rhythm.numberFrames")),
      //                           "frameHop",      int(options.value<Real>("rhythm.frameHop")));

      connect(eqloudSource, rhythmExtractor->input("signal"));
      connect(rhythmExtractor->output("ticks"),        eqloudPool, rhythmspace + "beats_position");
      connect(rhythmExtractor->output("bpm"),          eqloudPool, rhythmspace + "bpm");
      connect(rhythmExtractor->output("estimates"),    eqloudPool, rhythmspace + "bpm_estimates");
      //connect(rhythmExtractor->output("rubatoStart"),  eqloudPool, rhythmspace + "rubato_start");
      //connect(rhythmExtractor->output("rubatoStop"),   eqloudPool, rhythmspace + "rubato_stop");
      connect(rhythmExtractor->output("bpmIntervals"), eqloudPool, rhythmspace + "bpm_intervals");
      // discard dummy value for confidence as 'degara' beat tracker is not 
      // able to compute it
      rhythmExtractor->output("confidence") >> NOWHERE; 

      // BPM Histogram descriptors
      Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
      connect(rhythmExtractor->output("bpmIntervals"), bpmhist->input("bpmIntervals"));
      connectSingleValue(bpmhist->output("firstPeakBPM"),     eqloudPool, rhythmspace + "first_peak_bpm");
      connectSingleValue(bpmhist->output("firstPeakWeight"),  eqloudPool, rhythmspace + "first_peak_weight");
      connectSingleValue(bpmhist->output("firstPeakSpread"),  eqloudPool, rhythmspace + "first_peak_spread");
      connectSingleValue(bpmhist->output("secondPeakBPM"),    eqloudPool, rhythmspace + "second_peak_bpm");
      connectSingleValue(bpmhist->output("secondPeakWeight"), eqloudPool, rhythmspace + "second_peak_weight");
      connectSingleValue(bpmhist->output("secondPeakSpread"), eqloudPool, rhythmspace + "second_peak_spread");

      // Onset Detection
      Algorithm* onset = factory.create("OnsetRate");
      connect(eqloudSource, onset->input("signal"));
      connect(onset->output("onsetTimes"), eqloudPool, rhythmspace + "onset_times");
      connect(onset->output("onsetRate"), NOWHERE ); //pool, rhythmspace + "onset_rate"); // this is done later
    }
  }

  cout << "Process step 2: Low Level" << endl;
  Network network(audio_2);
  network.run();

  if (!shortsound) {
    // check if we processed enough audio for it to be useful, in particular did
    // we manage to get an estimation for the loudness (2 seconds required)
    try {
      if (eqloud) eqloudPool.value<vector<Real> >(llspace + "loudness")[0];
      else neqloudPool.value<vector<Real> >(llspace + "loudness")[0];
    }
    catch (EssentiaException&) {
      cout << "ERROR: File is too short (< 2sec)... Aborting..." << endl;
      exit(6);
    }
  }

  if (options.value<Real>("rhythm.compute") != 0) {
    // compute onset rate = len(onsets) / len(audio)
    if (neqloud) {
      neqloudPool.set(rhythmspace + "onset_rate", neqloudPool.value<vector<Real> >(rhythmspace + "onset_times").size()
         / (Real)audio_2->output("audio").totalProduced()
         * neqloudPool.value<Real>("metadata.audio_properties.analysis_sample_rate"));
    }
    if (eqloud) {
      eqloudPool.set(rhythmspace + "onset_rate", eqloudPool.value<vector<Real> >(rhythmspace + "onset_times").size()
                     / (Real)audio_2->output("audio").totalProduced()
                     * eqloudPool.value<Real>("metadata.audio_properties.analysis_sample_rate"));
    }
  }


  // delete network only now, because we needed audio_2->output("audio") to
  // compute the onset rate on the previous line.
  //deleteNetwork(audio_2);
}

// outdated beat tracker
//void computeBeatTrack(Pool& pool, const Pool& options, const string& nspace) {
//
//  /*************************************************************************
//   *    3rd pass: Beat Track                                               *
//   *************************************************************************/
//
//  cout << "Process step 3: Beat Track" << endl;
//  BeatTrack(pool, options, nspace);
//}

void computeMidLevel(const string& audioFilename, Pool& neqloudPool,
                     Pool& eqloudPool, const Pool& options,
                     Real startTime, Real endTime, const string& nspace) {

  /*************************************************************************
   *    4th pass: HPCP & beats loudness (depend on some descriptors that   *
   *              have been computed during the 2nd pass)                  *
   *************************************************************************/
  Real analysisSampleRate = options.value<Real>("analysisSampleRate");
  Real replayGain = 0;
  string downmix = "mix";

  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud  = options.value<Real>("equalLoudness")  != 0;

  if (eqloud) {
    replayGain = eqloudPool.value<Real>("metadata.audio_properties.replay_gain");
    downmix = eqloudPool.value<string>("metadata.audio_properties.downmix");
  }
  else {
    replayGain = neqloudPool.value<Real>("metadata.audio_properties.replay_gain");
    downmix = neqloudPool.value<string>("metadata.audio_properties.downmix");
  }

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_3 = factory.create("EasyLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime",  startTime,
                                      "endTime",    endTime,
                                      "replayGain", replayGain,
                                      "downmix",    downmix);


  if (neqloud) {
    SourceBase& neqloudSource = audio_3->output("audio");
    // Compute Tonal descriptors (needed TuningFrequency before)
    if (options.value<Real>("tonal.compute") != 0)
      TonalDescriptors(neqloudSource, neqloudPool, options, nspace);


    // Compute the loudness at the beats position (needed beats position)
    if (options.value<Real>("rhythm.compute") != 0) {
      string rhythmspace = "rhythm.";
      if (!nspace.empty()) rhythmspace = nspace + ".rhythm.";
      vector<Real> ticks = neqloudPool.value<vector<Real> >(rhythmspace + "beats_position");

      Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                                "sampleRate", analysisSampleRate,
                                                "beats", ticks);

      connect(neqloudSource, beatsLoudness->input("signal"));
      connect(beatsLoudness->output("loudness"), neqloudPool, rhythmspace + "beats_loudness");
      connect(beatsLoudness->output("loudnessBandRatio"), neqloudPool, rhythmspace + "beats_loudness_band_ratio");
    }
  }
  if (eqloud) {
    Algorithm* eqloud3 = factory.create("EqualLoudness");
    connect(audio_3->output("audio"), eqloud3->input("signal"));
    SourceBase& eqloudSource = eqloud3->output("signal");
    // Compute Tonal descriptors (needed TuningFrequency before)
    if (options.value<Real>("tonal.compute") != 0)
      TonalDescriptors(eqloudSource, eqloudPool, options, nspace);


    // Compute the loudness at the beats position (needed beats position)
    if (options.value<Real>("rhythm.compute") != 0) {
      string rhythmspace = "rhythm.";
      if (!nspace.empty()) rhythmspace = nspace + ".rhythm.";
      vector<Real> ticks = eqloudPool.value<vector<Real> >(rhythmspace + "beats_position");

      Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                                "sampleRate", analysisSampleRate,
                                                "beats", ticks);

      connect(eqloudSource, beatsLoudness->input("signal"));
      connect(beatsLoudness->output("loudness"), eqloudPool, rhythmspace + "beats_loudness");
      connect(beatsLoudness->output("loudnessBandRatio"), eqloudPool, rhythmspace + "beats_loudness_band_ratio");
    }
  }
  cout << "Process step 4: Mid Level" << endl;
  Network network(audio_3);
  network.run();
}

void computePanning(const string& audioFilename, Pool& neqloudPool,
                    Pool& eqloudPool, const Pool& options,
                    Real startTime, Real endTime, const string& nspace) {

  /*************************************************************************
   *    5th pass: Panning                                                  *
   *                                                                       *
   *************************************************************************/
  bool neqloud = options.value<Real>("nequalLoudness") != 0;
  bool eqloud =  options.value<Real>("equalLoudness")  != 0;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_4 = factory.create("AudioLoader",
                                      "filename", audioFilename);
  Algorithm* trimmer = factory.create("Trimmer",
                                      "startTime", startTime,
                                      "endTime", endTime);
  SourceBase& audioSource_3 = trimmer->output("signal");
  connect(audio_4->output("audio"), trimmer->input("signal"));
  connect(audio_4->output("sampleRate"), NOWHERE);
  connect(audio_4->output("numberChannels"), NOWHERE);
  // no difference between eqloud and neqloud, both are taken as non eqloud
  if (neqloud) Panning(audioSource_3, neqloudPool, options, nspace);
  if (eqloud) Panning(audioSource_3, eqloudPool, options, nspace);
  cout << "Process step 5: Panning" << endl;
  Network network(audio_4);
  network.run();
}

void computeHighlevel(Pool& pool, const Pool& options, const string& nspace) {

  /*************************************************************************
   *    6th pass: High-level descriptors that depend on others, but we     *
   *              don't need to stream the audio anymore                   *
   *************************************************************************/

  cout << "Process step 5: High Level" << endl;

  // Average Level
  if (options.value<Real>("average_loudness.compute") != 0)
    LevelAverage(pool, nspace);

  // SFX Descriptors
  if (options.value<Real>("sfx.compute") != 0)
    SFXPitch(pool, nspace);

  // Tuning System Features
  if (options.value<Real>("tonal.compute") != 0) {
    TuningSystemFeatures(pool, nspace);
    // Pool Cleaning (remove temporary descriptors)
    TonalPoolCleaning(pool, nspace);
  }

  // Add missing descriptors which are not computed yet, but will be for the
  // final release or during the 1.x cycle. However, the schema need to be
  // complete before that, so just put default values for these.
  PostProcess(pool, options, nspace);
}

Pool computeAggregation(Pool& pool, const Pool& options, int nSegments) {

  // choose which descriptors stats to output
  const char* defaultStats[] = { "mean", "var", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };

  map<string, vector<string> > exceptions;
  const vector<string>& descNames = pool.descriptorNames();
  for (int i=0; i<(int)descNames.size(); i++) {
    if (descNames[i].find("lowlevel.mfcc") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("lowlevel.mfccStats");
      continue;
    }
    if (descNames[i].find("lowlevel.") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("lowlevel.stats");
      continue;
    }
    if (descNames[i].find("rhythm.") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("rhythm.stats");
      continue;
    }
    if (descNames[i].find("tonal.") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("tonal.stats");
      continue;
    }
    if (descNames[i].find("sfx.") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("sfx.stats");
      continue;
    }
    if (descNames[i].find("panning.") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("panning.stats");
      continue;
    }
  }

  // in case there is segmentation:
  if (nSegments > 0) {
    vector<string> value(1, "copy");
    exceptions["segmentation.timestamps"] = value;
  }

  standard::Algorithm* aggregator = standard::AlgorithmFactory::create("PoolAggregator",
                                                                       "defaultStats", arrayToVector<string>(defaultStats),
                                                                       "exceptions", exceptions);
  Pool poolStats;
  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);

  cout << "Process step 6: Aggregation" << endl;

  aggregator->compute();

  delete aggregator;

  return poolStats;
}

void outputToFile(Pool& pool, const string& outputFilename, const Pool& options) {

  cout << "Writing results to file " << outputFilename << endl;
  // some descriptors depend on lowlevel descriptors but it might be that the
  // config file was set lowlevel.compute: false. In this case, the ouput yaml
  // file should not contain lowlevel features. The rest of namespaces should
  // only be computed if they were set explicitly in the config file
  if (options.value<Real>("lowlevel.compute") == 0) pool.removeNamespace("lowlevel");

  // TODO: merge results pool with options pool so configuration is also
  // available in the output file
  mergeOptionsAndResults(pool, options);

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "doubleCheck", true,
                                                                   "format", (options.value<Real>("outputJSON") != 0) ? "json" : "yaml");
  output->input("pool").set(pool);
  output->compute();

  delete output;
}


void addSVMDescriptors(Pool& pool) {
  cout << "Process step 7: SVM Models" << endl;
  //const char* svmModels[] = {}; // leave this empty if you don't have any SVM models
  const char* svmModels[] = { "genre_tzanetakis", "genre_dortmund",
                              "genre_electronica", "genre_rosamerica",
                              "mood_acoustic", "mood_aggressive",
                              "mood_electronic", "mood_happy",
                              "mood_party", "mood_relaxed", "mood_sad",
                              "perceptual_speed", "timbre",
                              "culture", "gender", "live_studio",
                              "mirex-moods", "ballroom",
                              "voice_instrumental", "speech_music"
  };

  string pathToSvmModels;

#ifdef OS_WIN32
  pathToSvmModels = "svm_models\\";
#else
  pathToSvmModels = "svm_models/";
#endif

  for (int i=0; i<(int)ARRAY_SIZE(svmModels); i++) {
    //cout << "adding HL desc: " << svmModels[i] << endl;
    string modelFilename = pathToSvmModels + string(svmModels[i]) + ".history";
    standard::Algorithm* svm = standard::AlgorithmFactory::create("GaiaTransform",
                                                                  "history", modelFilename);

    svm->input("pool").set(pool);
    svm->output("pool").set(pool);
    svm->compute();

    delete svm;
  }
}
