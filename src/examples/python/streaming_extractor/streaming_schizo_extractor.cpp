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

#include <algorithm>
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

void computeSegments(const string& audioFilename, Real startTime, Real endTime,
                     Pool& eqPool, Pool& neqPool);
void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& eqPool, Pool& neqPool);

void computeReplayGain(const string& audioFilename, Real startTime, Real endTime, Pool& eqPool);
void computeLowLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& eqPool, Pool& neqPool, const string& nspace = "");
void computeMidLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& eqPool, Pool& neqPool, const string& nspace = "");
void computePanning(const string& audioFilename, Real startTime, Real endTime,
                    Pool& eqPool, Pool& neqPool, const string& nspace = "");
void computeHighlevel(Pool& eqPool, const string& nspace = "");
Pool computeAggregation(Pool& eqPool, int segments=0);
void addSVMDescriptors(Pool& eqPool, Pool& neqPool);
void outputToFile(Pool& eqPool, const string& outputFilename);

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
  // pool for storing equal loudness results
  Pool eqPool;
  computeReplayGain(audioFilename, startTime, endTime, eqPool);
  // non equal loudness pool:
  Pool neqPool;
  neqPool.merge(eqPool, "replace");

  eqPool.set("metadata.audio_properties.equal_loudness", true);
  neqPool.set("metadata.audio_properties.equal_loudness", false);

  if (computeSegmentation) {
    // pool for storing segments:
    computeSegments(audioFilename, startTime, endTime, eqPool, neqPool);
    vector<Real> segments = eqPool.value<vector<Real> >("segmentation.timestamps");
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
      eqPool.set(ns.str(), sn);
      neqPool.set(ns.str(), sn);

      // set segment scope
      ns.str(""); ns << "segments." << sn << ".scope";
      vector<Real> scope(2, 0);
      scope[0] = start;
      scope[1] = end;
      eqPool.set(ns.str(), scope);
      neqPool.set(ns.str(), scope);

      // compute descriptors
      ns.str(""); ns << "segments.segment_" << i << ".descriptors";

      computeLowLevel(audioFilename, start, end, eqPool, neqPool, ns.str());
      computeMidLevel(audioFilename, start, end, eqPool, neqPool, ns.str());
      //computePanning(audioFilename, start, end, eqPool, neqPool, ns.str());
      computeHighlevel(eqPool, ns.str());
      computeHighlevel(neqPool, ns.str());
    }
    if (endTime > eqPool.value<Real>("metadata.audio_properties.length")) {
      endTime = eqPool.value<Real>("metadata.audio_properties.length");
    }
    cout << "\n**************************************************************************\n";
    //cout << "\n compute the rest of descriptors for the entire audio from " << startTime << "s to " << endTime << "s";
    //cout << "\n**************************************************************************" << endl;
    //computeLowLevel(audioFilename, startTime, endTime, eqPool, neqPool); // already computed when performing segmentation
    computeMidLevel(audioFilename, startTime, endTime, eqPool, neqPool);
    computeHighlevel(eqPool);
    computeHighlevel(neqPool);

    Pool stats = computeAggregation(eqPool, segments.size());
    // Add this line when svm models are trained
    //addSVMDescriptors(stats);
    outputToFile(stats, outputFilename);
    stats.clear();
    stats = computeAggregation(neqPool, segments.size());
    string baseFilename = outputFilename.substr(0, outputFilename.rfind('.'));
    string neqOutputFilename = baseFilename + "..neq.sig";
    outputToFile(stats, neqOutputFilename);
  }
  else {
    try {
      compute(audioFilename, outputFilename, startTime, endTime, eqPool, neqPool);
    }
    catch (EssentiaException& e) {
      cout << e.what() << endl;
      throw;
    }
  }

  eqPool.remove("metadata.audio_properties.downmix");
  neqPool.remove("metadata.audio_properties.downmix");
  essentia::shutdown();

  return 0;
}

void compute(const string& audioFilename, const string& outputFilename,
             Real startTime, Real endTime, Pool& eqPool, Pool& neqPool) {
  computeLowLevel(audioFilename, startTime, endTime, eqPool, neqPool);
  computeMidLevel(audioFilename, startTime, endTime, eqPool, neqPool);
  //computePanning(audioFilename, startTime, endTime, eqPool, neqPool);
  computeHighlevel(eqPool);
  computeHighlevel(neqPool);
  Pool stats = computeAggregation(eqPool);
  // Add this line when svm models are trained
  //addSVMDescriptors(stats);
  outputToFile(stats, outputFilename);
  stats.clear();
  stats = computeAggregation(neqPool);

  string baseFilename = outputFilename.substr(0, outputFilename.rfind('.'));
  string neqOutputFilename = baseFilename + "..neq.sig";
  outputToFile(stats, neqOutputFilename);

}

void computeSegments(const string& audioFilename, Real startTime, Real endTime, Pool& eqPool, Pool& neqPool) {

  int lowlevelHopSize = 1024;
  int minimumSegmentsLength = 10;
  int size1 = 1000, inc1 = 300, size2 = 600, inc2 = 50, cpw = 5;

  // compute low level features to feed SBIc
  computeLowLevel(audioFilename, startTime, endTime, eqPool, neqPool);

  vector<vector<Real> > features;
  try {
    features = eqPool.value<vector<vector<Real> > >("lowlevel.mfcc");
  }
  catch(const EssentiaException&) {
    cout << "Error: could not find MFCC features in low level eqPool. Aborting..." << endl;
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
    analysisSampleRate = eqPool.value<Real>("metadata.audio_properties.analysis_sample_rate");
  }
  catch(const EssentiaException&) {
    throw EssentiaException("Warning: StreamingExtractor::computeSegments, could not find analysis sampling rate");
  }
  for (int i=0; i<int(segments.size()); ++i) {
    segments[i] *= Real(lowlevelHopSize)/analysisSampleRate;
    eqPool.add("segmentation.timestamps", segments[i]);
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

void computeLowLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& eqPool, Pool& neqPool, const string& nspace) {


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
  getAnalysisData(eqPool, replayGain, analysisSampleRate, downmix);

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_2 = factory.create("EasyLoader",
                                      "filename", audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime", startTime,
                                      "endTime", endTime,
                                      "replayGain", replayGain,
                                      "downmix", downmix);

  Algorithm* eqloud = factory.create("EqualLoudness");
  connect(audio_2->output("audio"), eqloud->input("signal"));

  SourceBase& neqloudSource = audio_2->output("audio");
  SourceBase& eqloudSource = eqloud->output("signal");

  // Low-Level Spectral Descriptors
  Algorithm * eqLowLevelSpectral = new LowLevelSpectralExtractor();
  Algorithm * neqLowLevelSpectral = new LowLevelSpectralExtractor();
  neqLowLevelSpectral->configure("frameSize", _lowlevelFrameSize,
                                 "hopSize", _lowlevelHopSize,
                                 "halfSampleRate", analysisSampleRate*0.5);

  eqLowLevelSpectral->configure("frameSize", _lowlevelFrameSize,
                                "hopSize", _lowlevelHopSize,
                                "halfSampleRate", analysisSampleRate*0.5);

  // connect inputs:
  connect(neqloudSource, neqLowLevelSpectral->input("signal"));
  connect(eqloudSource, eqLowLevelSpectral->input("signal"));
  // connect outputs:
  const char * sfxDescArray[] = {"inharmonicity", "oddtoevenharmonicenergyratio", "tristimulus"};
  vector<string> sfxDesc = arrayToVector<string>(sfxDescArray);
  OutputMap::const_iterator it = eqLowLevelSpectral->outputs().begin();
  for (; it != eqLowLevelSpectral->outputs().end(); ++it) {
    string output_name = it->first;
    string ns = llspace; // namespace
    if (find(sfxDesc.begin(), sfxDesc.end(), output_name) != sfxDesc.end()) {
      ns = sfxspace;
    }
    connect(*it->second, eqPool, ns + output_name);
  }
  it = neqLowLevelSpectral->outputs().begin();
  for (; it != neqLowLevelSpectral->outputs().end(); ++it) {
    string output_name = it->first;
    string ns = llspace; // namespace
    if (find(sfxDesc.begin(), sfxDesc.end(), output_name) != sfxDesc.end()) {
      ns = sfxspace;
    }
    connect(*it->second, neqPool, ns + output_name);
  }
  // alternatively:
  // connect(lowLevelSpectral->output("silence_rate_20dB"), eqPool, llspace + "silence_rate_20dB");
  // connect(lowLevelSpectral->output("silence_rate_30dB"), eqPool, llspace + "silence_rate_30dB");
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
    connect(*it->second, eqPool, llspace + it->first);
    connect(*it->second, neqPool, llspace + it->first);
  }

  // Level Descriptor
  Algorithm * level = new LevelExtractor();
  level->configure("frameSize", _dynamicsFrameSize,
                   "hopSize", _dynamicsHopSize);

  connect(eqloudSource, level->input("signal"));
  connect(level->output("loudness"), eqPool, llspace + "loudness");
  connect(level->output("loudness"), neqPool, llspace + "loudness");

  // Tuning Frequency
  Algorithm * eqTuningFrequency = new TuningFrequencyExtractor();
  eqTuningFrequency->configure("frameSize", _tonalFrameSize,
                             "hopSize", _tonalHopSize);

  connect(eqloudSource, eqTuningFrequency->input("signal"));
  connect(eqTuningFrequency->output("tuningFrequency"), eqPool, tonalspace + "tuning_frequency");

  Algorithm * neqTuningFrequency = new TuningFrequencyExtractor();
  neqTuningFrequency->configure("frameSize", _tonalFrameSize,
                             "hopSize", _tonalHopSize);

  connect(neqloudSource, neqTuningFrequency->input("signal"));
  connect(neqTuningFrequency->output("tuningFrequency"), neqPool, tonalspace + "tuning_frequency");

  // Rhythm descriptors & BPM Histogram descriptors:
  Algorithm * eqRhythmExtractor = new RhythmDescriptorsExtractor();
  eqRhythmExtractor->configure();

  connect(eqloudSource, eqRhythmExtractor->input("signal"));
  it = eqRhythmExtractor->outputs().begin();
  for (; it != eqRhythmExtractor->outputs().end(); ++it) {
    connect(*it->second, eqPool, rhythmspace + it->first);
  }

  Algorithm * neqRhythmExtractor = new RhythmDescriptorsExtractor();
  neqRhythmExtractor->configure();

  connect(neqloudSource, neqRhythmExtractor->input("signal"));
  it = neqRhythmExtractor->outputs().begin();
  for (; it != neqRhythmExtractor->outputs().end(); ++it) {
    connect(*it->second, neqPool, rhythmspace + it->first);
  }

  // Onset Detection
  Algorithm* eqOnset = factory.create("OnsetRate");
  connect(eqloudSource, eqOnset->input("signal"));
  connect(eqOnset->output("onsetTimes"), eqPool, rhythmspace + "onset_times");
  connect(eqOnset->output("onsetRate"), NOWHERE );

  Algorithm* neqOnset = factory.create("OnsetRate");
  connect(neqloudSource, neqOnset->input("signal"));
  connect(neqOnset->output("onsetTimes"), neqPool, rhythmspace + "onset_times");
  connect(neqOnset->output("onsetRate"), NOWHERE );

  cout << "Process step 2: Low Level" << endl;
  runGenerator(audio_2);

  // check if we processed enough audio for it to be useful, in particular did
  // we manage to get an estimation for the loudness (2 seconds required)
  try {
    eqPool.value<vector<Real> >(llspace + "loudness")[0];
  }
  catch (const EssentiaException& e) {
    cout << "ERROR: File is too short (< 2sec)... Aborting..." << endl;
    exit(6);
  }

  // compute onset rate = len(onsets) / len(audio)
  eqPool.set(rhythmspace + "onset_rate", eqPool.value<vector<Real> >(rhythmspace + "onset_times").size()
     / (Real)audio_2->output("audio").totalProduced()
     * eqPool.value<Real>("metadata.audio_properties.analysis_sample_rate"));

  neqPool.set(rhythmspace + "onset_rate", neqPool.value<vector<Real> >(rhythmspace + "onset_times").size()
     / (Real)audio_2->output("audio").totalProduced()
     * neqPool.value<Real>("metadata.audio_properties.analysis_sample_rate"));

  // delete network only now, because we needed audio_2->output("audio") to
  // compute the onset rate on the previous line.
  deleteNetwork(audio_2);
}

void computeMidLevel(const string& audioFilename, Real startTime, Real endTime,
                     Pool& eqPool, Pool& neqPool, const string& nspace) {

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
  getAnalysisData(eqPool, replayGain, analysisSampleRate, downmix);

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio_3 = factory.create("EasyLoader",
                                      "filename", audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime", startTime,
                                      "endTime", endTime,
                                      "replayGain", replayGain,
                                      "downmix", downmix);
  Algorithm* eqloud = factory.create("EqualLoudness");
  connect(audio_3->output("audio"), eqloud->input("signal"));
  SourceBase& neqloudSource_2 = audio_3->output("audio");
  SourceBase& eqloudSource_2 = eqloud->output("signal");

  // Compute Tonal descriptors (needed TuningFrequency before)
  Real eqTuningFreq = eqPool.value<vector<Real> >(tonalspace + "tuning_frequency").back();
  Algorithm * eqTonalDescriptors = new TonalDescriptorsExtractor();
  eqTonalDescriptors->configure("frameSize", _tonalFrameSize,
                                "hopSize", _tonalHopSize,
                                "tuningFrequency", eqTuningFreq);
  connect(eqloudSource_2, eqTonalDescriptors->input("signal"));
  OutputMap::const_iterator it = eqTonalDescriptors->outputs().begin();
  for (; it != eqTonalDescriptors->outputs().end(); ++it) {
    connect(*it->second, eqPool, tonalspace + it->first);
  }

  Real neqTuningFreq = neqPool.value<vector<Real> >(tonalspace + "tuning_frequency").back();
  Algorithm * neqTonalDescriptors = new TonalDescriptorsExtractor();
  neqTonalDescriptors->configure("frameSize", _tonalFrameSize,
                                 "hopSize", _tonalHopSize,
                                 "tuningFrequency", neqTuningFreq);
  connect(neqloudSource_2, neqTonalDescriptors->input("signal"));
  it = neqTonalDescriptors->outputs().begin();
  for (; it != neqTonalDescriptors->outputs().end(); ++it) {
    connect(*it->second, neqPool, tonalspace + it->first);
  }

  // Compute the loudness at the beats position (needed beats position)
  vector<Real> eqTicks = eqPool.value<vector<Real> >(rhythmspace + "beats_position");
  vector<Real> neqTicks = neqPool.value<vector<Real> >(rhythmspace + "beats_position");

  Algorithm* eqBeatsLoudness = factory.create("BeatsLoudness",
                                              "sampleRate", analysisSampleRate,
                                              "beats", eqTicks);

  connect(eqloudSource_2, eqBeatsLoudness->input("signal"));
  connect(eqBeatsLoudness->output("loudness"),     eqPool, rhythmspace + "beats_loudness");
  connect(eqBeatsLoudness->output("loudnessBass"), eqPool, rhythmspace + "beats_loudness_bass");

  Algorithm* neqBeatsLoudness = factory.create("BeatsLoudness",
                                               "sampleRate", analysisSampleRate,
                                               "beats", neqTicks);

  connect(neqloudSource_2, neqBeatsLoudness->input("signal"));
  connect(neqBeatsLoudness->output("loudness"),     neqPool, rhythmspace + "beats_loudness");
  connect(neqBeatsLoudness->output("loudnessBass"), neqPool, rhythmspace + "beats_loudness_bass");

  cout << "Process step 3: Mid Level" << endl;
  runGenerator(audio_3);
  deleteNetwork(audio_3);
}

void computePanning(const string& audioFilename, Real startTime, Real endTime,
                    Pool& eqPool, Pool& neqPool, const string& nspace) {

  /*************************************************************************
   *    4th pass: Panning                                                  *
   *                                                                       *
   *************************************************************************/
  Real analysisSampleRate = 44100.0;
  try {
    analysisSampleRate = eqPool.value<Real>("metadata.audio_properties.analysis_sample_rate");
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
  connect(panningExtractor->output("panning_coefficients"), eqPool, "panning_coefficients");
  connect(panningExtractor->output("panning_coefficients"), neqPool, "panning_coefficients");

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
