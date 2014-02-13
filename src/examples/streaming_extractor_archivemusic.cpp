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

// Streaming extractor designed for analysis of music collections on Archive.org

#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiamath.h>
#include <essentia/essentiautil.h>
#include <essentia/scheduler/network.h>

// helper functions
#include "streaming_extractorutils.h"
#include "streaming_extractorlowlevel.h"
#include "streaming_extractorsfx.h"
#include "streaming_extractortonal.h"
#include "streaming_extractorpostprocess.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

void compute(const string& audioFilename, const string& outputFilename, Pool& pool, const Pool& options);
void computeReplayGain(const string& audioFilename, Pool& pool, const Pool& options);
void computeStep2(const string& audioFilename, Pool& pool, const Pool& options, Real startTime, Real endTime);
void computeStep3(const string& audioFilename, Pool& pool, const Pool& options, Real startTime, Real endTime);
void computeStep4(Pool& pool, const Pool& options);
Pool computeAggregation(Pool& pool, const Pool& options, int segments=0);
void addSVMDescriptors(Pool& pool);
void outputToFile(Pool& pool, const string& outputFilename, bool outputJSON);
void setExtractorOptions(Pool& pool);

void usage() {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: streaming_extractor_archivemusic input_audiofile output_textfile" << endl;
    exit(1);
}


int main(int argc, char* argv[]) {

  string audioFilename, outputFilename;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    case 4: // profile supplied
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    default:
      usage();
  }

  // Register the algorithms in the factory(ies)
  essentia::init();

  // pool cotaining profile (configure) options:
  Pool options;

  // set configuration from file or otherwise use default settings:
  setExtractorOptions(options);

  // pool for storing results
  Pool pool; // non equal loudness pool

  try {
    compute(audioFilename, outputFilename, pool, options);
  }
  catch (EssentiaException& e) {
    cout << e.what() << endl;
    throw;
  }

  essentia::shutdown();

  return 0;
}

void compute(const string& audioFilename, const string& outputFilename, Pool& pool, const Pool& options) {
  
  // add extractor version
  pool.set("metadata.version.extractor", "archive-music 1.0");
  
  pool.set("metadata.audio_properties.equal_loudness", false);

  // compute features for the whole song
  computeReplayGain(audioFilename, pool, options);
  Real startTime = options.value<Real>("startTime");
  Real endTime = options.value<Real>("endTime");

  if (endTime > pool.value<Real>("metadata.audio_properties.length")) {
      endTime = pool.value<Real>("metadata.audio_properties.length");
  }

  computeStep2(audioFilename, pool, options, startTime, endTime);
  computeStep3(audioFilename, pool, options, startTime, endTime);
  computeStep4(pool, options);
  
  Pool stats = computeAggregation(pool, options);

#if HAVE_GAIA2
  addSVMDescriptors(stats);
#else
  cout << "Warning: Essentia was compiled without Gaia2 library, skipping process step 6 (cannot compute SVM models)" << endl;
#endif
  
  outputToFile(stats, outputFilename, true);
  return;
}


void computeReplayGain(const string& audioFilename, Pool& pool, const Pool& options) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Real analysisSampleRate = options.value<Real>("analysisSampleRate");


  /*************************************************************************
   *    1st pass: get metadata and replay gain                             *
   *************************************************************************/

  readMetadata(audioFilename, pool);
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
    if (replayGain > 40.0) { // before it was set to 20 but it was found too conservative
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

void computeStep2(const string& audioFilename, Pool& pool,
                     const Pool& options, Real startTime, Real endTime) {
  /*************************************************************************
   *    2nd pass: normalize the audio with replay gain, compute as         *
   *              many lowlevel/rhythm descriptors as possible             *
   *************************************************************************/
  
  string llspace = "lowlevel.";
  string rhythmspace = "rhythm.";

  Real replayGain = 0;
  string downmix = "mix";
  replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");
  downmix = pool.value<string>("metadata.audio_properties.downmix");
  
  Real analysisSampleRate = options.value<Real>("analysisSampleRate");

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

  LowLevelSpectral(neqloudSource, pool, options);

  // Low-Level Spectral Equal Loudness Descriptors
  // expects the audio source to already be equal-loudness filtered, so it
  // must use the eqloudSouce instead of neqloudSource
  LowLevelSpectralEqLoud(eqloudSource, pool, options);

  // Compute loudness using non-eqloud signal. Note that we could also use eqloud
  // signal (as did the extractors in Essentia 1.3) that would correspond to a 
  // "perceptual loudness". However, we opt for non-eqloud, as we can expect
  // that the performance  won't change much and there are better perceptual 
  // loudness estimation techniques.
  
  Level(neqloudSource, pool, options);

  // Tuning Frequency
  TuningFrequency(neqloudSource, pool, options);

  // Rhythm descriptors
  Algorithm* rhythmExtractor = factory.create("RhythmExtractor2013");
  rhythmExtractor->configure("method", options.value<string>("rhythm.method"),
                             "maxTempo", (int) options.value<Real>("rhythm.maxTempo"),
                             "minTempo", (int) options.value<Real>("rhythm.minTempo"));

  neqloudSource >> rhythmExtractor->input("signal");
  connect(rhythmExtractor->output("ticks"),        pool, rhythmspace + "beats_position");
  rhythmExtractor->output("confidence") >> NOWHERE; // dummy descriptor as 'degara' method does not estimate confidence
  connect(rhythmExtractor->output("bpm"),          pool, rhythmspace + "bpm");
  // NOTE: we do not need bpm estimates and intervals in the pool because
  //       they can be deduced from ticks position and occupy too much space
  connect(rhythmExtractor->output("estimates"),    NOWHERE);

  // BPM Histogram descriptors
  Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
  rhythmExtractor->output("bpmIntervals") >> bpmhist->input("bpmIntervals");
  connectSingleValue(bpmhist->output("firstPeakBPM"),     pool, rhythmspace + "bpm_histogram_first_peak");
  connectSingleValue(bpmhist->output("firstPeakWeight"),  pool, rhythmspace + "bpm_histogram_first_peak_weight");
  connectSingleValue(bpmhist->output("firstPeakSpread"),  pool, rhythmspace + "bpm_histogram_first_peak_spread");
  connectSingleValue(bpmhist->output("secondPeakBPM"),    pool, rhythmspace + "bpm_histogram_second_peak");
  connectSingleValue(bpmhist->output("secondPeakWeight"), pool, rhythmspace + "bpm_histogram_second_peak_weight");
  connectSingleValue(bpmhist->output("secondPeakSpread"), pool, rhythmspace + "bpm_histogram_second_peak_spread");

  // Onset Detection
  Algorithm* onset = factory.create("OnsetRate");
  neqloudSource >> onset->input("signal");
  connect(onset->output("onsetTimes"), pool, rhythmspace + "onset_times");
  connect(onset->output("onsetRate"), NOWHERE ); //pool, rhythmspace + "onset_rate"); // this is done later

  // Dynamic complexity
  Algorithm* dc = factory.create("DynamicComplexity");
  neqloudSource >> dc->input("signal");
  connect(dc->output("dynamicComplexity"), pool, llspace + "dynamic_complexity");
  dc->output("loudness") >> NOWHERE;

  // Danceability
  Algorithm* danceability = factory.create("Danceability");
  neqloudSource >> danceability->input("signal");
  connect(danceability->output("danceability"), pool, rhythmspace + "danceability");

  
  cout << "Process step 2: Low Level / Rhythm / Tonal" << endl;
  Network network(audio_2);
  network.run();

  // check if we processed enough audio for it to be useful, in particular did
  // we manage to get an estimation for the loudness (2 seconds required)
  try {
    pool.value<vector<Real> >(llspace + "loudness")[0];
  }
  catch (EssentiaException&) {
    cout << "ERROR: File is too short (< 2sec)... Aborting..." << endl;
    exit(6);
  }

  // compute average Level
  LevelAverage(pool);

  // compute onset rate = len(onsets) / len(audio)
  // we do not need onset times, as they are most probably incorrect, while onset_rate is more informative
  pool.set(rhythmspace + "onset_rate", pool.value<vector<Real> >(rhythmspace + "onset_times").size()
     / (Real)audio_2->output("audio").totalProduced()
     * pool.value<Real>("metadata.audio_properties.analysis_sample_rate"));
  pool.remove(rhythmspace + "onset_times");
}


void computeStep3(const string& audioFilename, Pool& pool, 
                     const Pool& options,Real startTime, Real endTime) {

  /*************************************************************************
   *    3nd pass: normalize the audio with replay gain, compute            *
   *              rhythm descriptors and tonal descriptors                 *
   *************************************************************************/
  string llspace = "lowlevel.";
  string rhythmspace = "rhythm.";
  
  Real analysisSampleRate = options.value<Real>("analysisSampleRate");
  Real replayGain = 0;
  string downmix = "mix";

  replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");
  downmix = pool.value<string>("metadata.audio_properties.downmix");

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_3 = factory.create("EasyLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime",  startTime,
                                      "endTime",    endTime,
                                      "replayGain", replayGain,
                                      "downmix",    downmix);

  SourceBase& neqloudSource = audio_3->output("audio");

  // Compute the loudness at the beats position (needed beats position)
  vector<Real> ticks = pool.value<vector<Real> >(rhythmspace + "beats_position");

  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                            "sampleRate", analysisSampleRate,
                                            "beats", ticks);
  neqloudSource >> beatsLoudness->input("signal");
  connect(beatsLoudness->output("loudness"), pool, rhythmspace + "beats_loudness");
  connect(beatsLoudness->output("loudnessBandRatio"), pool, rhythmspace + "beats_loudness_band_ratio");


  // Compute Tonal descriptors (needed TuningFrequency before)
  TonalDescriptors(neqloudSource, pool, options);

  cout << "Process step 3: Rhythm / Tonal" << endl;
  Network network(audio_3);
  network.run();

  // Tuning System Features
  TuningSystemFeatures(pool);
}


void computeStep4(Pool& pool, const Pool& options) {

  /*************************************************************************
   *    5th pass: Pool cleaning (remove temporary descriptors , but we     *
   *************************************************************************/

  cout << "Process step 4: Post-processing" << endl;

  TonalPoolCleaning(pool);
  PostProcess(pool, options);
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
    if (descNames[i].find("lowlevel.gfcc") != string::npos) {
      exceptions[descNames[i]] = options.value<vector<string> >("lowlevel.gfccStats");
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

  cout << "Process step 5: Aggregation" << endl;

  aggregator->compute();

  delete aggregator;

  return poolStats;
}

void outputToFile(Pool& pool, const string& outputFilename, bool outputJSON) {

  cout << "Writing results to file " << outputFilename << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "doubleCheck", true,
                                                                   "format", outputJSON ? "json" : "yaml");
  output->input("pool").set(pool);
  output->compute();
  delete output;
}


void addSVMDescriptors(Pool& pool) {
  cout << "Process step 6: SVM Models" << endl;
  //const char* svmModels[] = {}; // leave this empty if you don't have any SVM models
  const char* svmModels[] = { "danceability",
                              "genre_dortmund", "genre_electronic",
                              "genre_rosamerica", "genre_tzanetakis",
                              "mirex_ballroom",
                              "mood_acoustic", "mood_aggressive",
                              "mood_electronic", "mood_happy",
                              "mood_party", "mood_relaxed", "mood_sad",
                              "moods_mirex", 
                              "tonal_atonal", "voice_instrumental",
                              "timbre", "culture", "gender"};

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


void setExtractorOptions(Pool& pool) {
  // general
  pool.set("startTime", 0);
  pool.set("endTime", 1e6);
  pool.set("analysisSampleRate", 44100.0);
  string silentFrames = "noise";
  int zeroPadding = 0;
  string windowType = "hann";

  // lowlevel
  pool.set("lowlevel.frameSize", 2048);
  pool.set("lowlevel.hopSize", 1024);
  pool.set("lowlevel.zeroPadding", zeroPadding);
  pool.set("lowlevel.windowType", "blackmanharris62");
  pool.set("lowlevel.silentFrames", silentFrames);

  // average_loudness
  pool.set("average_loudness.frameSize", 88200);
  pool.set("average_loudness.hopSize", 44100);
  pool.set("average_loudness.windowType", windowType);
  pool.set("average_loudness.silentFrames", silentFrames);

  // rhythm
  pool.set("rhythm.method", "degara");
  pool.set("rhythm.minTempo", 40);
  pool.set("rhythm.maxTempo", 208);

  // tonal
  pool.set("tonal.frameSize", 4096);
  pool.set("tonal.hopSize", 2048);
  pool.set("tonal.zeroPadding", zeroPadding);
  pool.set("tonal.windowType", "blackmanharris62");
  pool.set("tonal.silentFrames", silentFrames);

  // stats
  const char* statsArray[] = { "mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };
  const char* mfccStatsArray[] = { "mean", "cov", "icov" };
  const char* gfccStatsArray[] = { "mean", "cov", "icov" };

  vector<string> stats = arrayToVector<string>(statsArray);
  vector<string> mfccStats = arrayToVector<string>(mfccStatsArray);
  vector<string> gfccStats = arrayToVector<string>(gfccStatsArray);
  for (int i=0; i<(int)stats.size(); i++) {
    pool.add("lowlevel.stats", stats[i]);
    pool.add("tonal.stats", stats[i]);
    pool.add("rhythm.stats", stats[i]);
    pool.add("sfx.stats", stats[i]);
  }
  for (int i=0; i<(int)mfccStats.size(); i++)
    pool.add("lowlevel.mfccStats", mfccStats[i]);
  for (int i=0; i<(int)gfccStats.size(); i++)
    pool.add("lowlevel.gfccStats", gfccStats[i]);
}
