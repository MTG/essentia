/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "freesoundextractor.h"

using namespace std;

namespace essentia {
namespace standard {

const char* FreesoundExtractor::name = "FreesoundExtractor";
const char* FreesoundExtractor::category = "Extractors";
const char* FreesoundExtractor::description = DOC("This algorithm is a wrapper for Freesound Extractor. See documentation for 'essentia_streaming_extractor_freesound'.");


FreesoundExtractor::FreesoundExtractor() {
  declareInput(_audiofile, "filename", "the input audiofile");
  declareOutput(_resultsStats, "results", "Analysis results pool with across-frames statistics");
  declareOutput(_resultsFrames, "resultsFrames", "Analysis results pool with computed frame values");
}


FreesoundExtractor::~FreesoundExtractor() {
  if (options.value<Real>("highlevel.compute")) {
#if HAVE_GAIA2
    if (_svms) delete _svms;
#endif  
  }
}


void FreesoundExtractor::reset() {}


void FreesoundExtractor::configure() {

  downmix = "mix";

  analysisSampleRate = parameter("analysisSampleRate").toReal();
  startTime = parameter("startTime").toReal();
  endTime = parameter("endTime").toReal();

  lowlevelFrameSize = parameter("lowlevelFrameSize").toInt();
  lowlevelHopSize = parameter("lowlevelHopSize").toInt();
  lowlevelZeroPadding = parameter("lowlevelZeroPadding").toInt();
  lowlevelSilentFrames = parameter("lowlevelSilentFrames").toLower();
  lowlevelWindowType = parameter("lowlevelWindowType").toLower();

  tonalFrameSize = parameter("tonalFrameSize").toInt();
  tonalHopSize = parameter("tonalHopSize").toInt();
  tonalZeroPadding = parameter("tonalZeroPadding").toInt();
  tonalSilentFrames = parameter("tonalSilentFrames").toLower();
  tonalWindowType = parameter("tonalWindowType").toLower();

  /*
  loudnessFrameSize = parameter("loudnessFrameSize").toInt();
  loudnessHopSize = parameter("loudnessHopSize").toInt();
  loudnessSilentFrames = parameter("loudnessSilentFrames").toLower();
  loudnessWindowType = parameter("loudnessWindowType").toLower();
  */
  
  rhythmMethod = parameter("rhythmMethod").toLower();
  rhythmMinTempo = parameter("rhythmMinTempo").toInt();
  rhythmMaxTempo = parameter("rhythmMaxTempo").toInt();

  lowlevelStats = parameter("lowlevelStats").toVectorString();
  tonalStats = parameter("tonalStats").toVectorString();
  rhythmStats = parameter("rhythmStats").toVectorString();
  mfccStats = parameter("mfccStats").toVectorString();
  gfccStats = parameter("gfccStats").toVectorString();

#if HAVE_GAIA2 
  if (parameter("highlevel").isConfigured()) { 
    svmModels = parameter("highlevel").toVectorString();
  }
#endif

  options.clear();
  setExtractorDefaultOptions();

  if (parameter("profile").isConfigured()) { 
    setExtractorOptions(parameter("profile").toString());

    analysisSampleRate = options.value<Real>("analysisSampleRate");
    startTime = options.value<Real>("startTime");
    endTime = options.value<Real>("endTime");
  }

  if (options.value<Real>("highlevel.compute")) {
#if HAVE_GAIA2 
    svmModels = options.value<vector<string> >("highlevel.svm_models");
    _svms = AlgorithmFactory::create("FreesoundExtractorSVM", "svms", svmModels);
#else
    E_WARNING("FreesoundExtractor: Gaia library is missing. Skipping configuration of SVM models.");
#endif
  }
}


void FreesoundExtractor::setExtractorDefaultOptions() {
  // general
  options.set("startTime", startTime);
  options.set("endTime", endTime);
  options.set("analysisSampleRate", analysisSampleRate);

  // lowlevel
  options.set("lowlevel.frameSize", lowlevelFrameSize);
  options.set("lowlevel.hopSize", lowlevelHopSize);
  options.set("lowlevel.zeroPadding", lowlevelZeroPadding);
  options.set("lowlevel.windowType", lowlevelWindowType);
  options.set("lowlevel.silentFrames", lowlevelSilentFrames);

  // tonal
  options.set("tonal.frameSize", tonalFrameSize);
  options.set("tonal.hopSize", tonalHopSize);
  options.set("tonal.zeroPadding", tonalZeroPadding);
  options.set("tonal.windowType", tonalWindowType);
  options.set("tonal.silentFrames", tonalSilentFrames);

  // average_loudness 
  // Note: below are parameters used in MusicExtractor, but 
  // in FreesoundExtractor we use the same parameters as for low-level.
  //options.set("average_loudness.frameSize", loudnessFrameSize);
  //options.set("average_loudness.hopSize", loudnessHopSize);
  //options.set("average_loudness.windowType", loudnessWindowType);
  //options.set("average_loudness.silentFrames", loudnessSilentFrames);

  // rhythm
  options.set("rhythm.method", rhythmMethod);
  options.set("rhythm.minTempo", rhythmMinTempo);
  options.set("rhythm.maxTempo", rhythmMaxTempo);

  // statistics
  options.set("lowlevel.stats", lowlevelStats); 
  options.set("tonal.stats", tonalStats);
  options.set("rhythm.stats", rhythmStats);

  options.set("lowlevel.mfccStats", mfccStats);
  options.set("lowlevel.gfccStats", gfccStats);

  // high-level
  options.set("highlevel.compute", false);
#if HAVE_GAIA2
  if (!svmModels.empty()) {
    options.add("highlevel.svm_models", svmModels);
    options.set("highlevel.compute", true);
  }
#endif
}


void FreesoundExtractor::compute() {
  const string& audioFilename = _audiofile.get();

  Pool& resultsStats = _resultsStats.get();
  Pool& resultsFrames = _resultsFrames.get();

  Pool results;
  Pool stats;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  results.set("metadata.version.essentia", essentia::version);
  results.set("metadata.version.essentia_git_sha", essentia::version_git_sha);
  results.set("metadata.version.extractor", FREESOUND_EXTRACTOR_VERSION);
  // TODO: extractor_build_id

  results.set("metadata.audio_properties.analysis.equal_loudness", false);
  results.set("metadata.audio_properties.analysis.sample_rate", analysisSampleRate);
  results.set("metadata.audio_properties.analysis.downmix", downmix);
  results.set("metadata.audio_properties.analysis.start_time", startTime);
  //results.set("metadata.audio_properties.analysis.end_time", endTime);

  // Reading metadata for Freesound too. This could be useful.  
  E_INFO("FreesoundExtractor: Read metadata");
  readMetadata(audioFilename, results);
 
  E_INFO("FreesoundExtractor: Compute md5 audio hash, codec, length, and EBU 128 loudness");
  computeAudioMetadata(audioFilename, results);
  
  //TODO: add algorithm option to compute with replay gain? 
  //E_INFO("FreesoundExtractor: Replay gain");
  //computeReplayGain(audioFilename, results);

  E_INFO("FreesoundExtractor: Compute audio features");

  streaming::Algorithm* loader = factory.create("EasyLoader",
                                    "filename",   audioFilename,
                                    "sampleRate", analysisSampleRate,
                                    "startTime",  startTime,
                                    "endTime",    endTime,
                                    //"replayGain", replayGain,
                                    "downmix",    downmix);

  FreesoundLowlevelDescriptors *lowlevel = new FreesoundLowlevelDescriptors(options);
  FreesoundRhythmDescriptors *rhythm = new FreesoundRhythmDescriptors(options);
  FreesoundTonalDescriptors *tonal = new FreesoundTonalDescriptors(options);
  FreesoundSfxDescriptors *sfx = new FreesoundSfxDescriptors(options);
 
  SourceBase& source = loader->output("audio");
  lowlevel->createNetwork(loader->output("audio"),results);
  rhythm->createNetwork(source, results);
  tonal->createNetwork(source, results);
  sfx->createNetwork(loader->output("audio"),results);
  sfx->createHarmonicityNetwork(loader->output("audio"), results);            

  scheduler::Network network(loader);
  network.run();
  
  // Descriptors that require values from other descriptors in the previous chain
  
  // requires 'loudness'
  lowlevel->computeAverageLoudness(results);

  streaming::Algorithm* loader_2 = factory.create("EasyLoader",
                                       "filename",   audioFilename,
                                       "sampleRate", analysisSampleRate,
                                       "startTime",  startTime,
                                       "endTime",    endTime,
                                       //"replayGain", replayGain,
                                       "downmix",    downmix);

  // requires 'beat_positions'
  rhythm->createNetworkBeatsLoudness(loader_2->output("audio"), results);  

  scheduler::Network network_2(loader_2);
  network_2.run();

  // requires 'pitch'
  vector<Real> pitch = results.value<vector<Real> >("lowlevel.pitch");
  VectorInput<Real> *pitchVector = new VectorInput<Real>();
  pitchVector->setVector(&pitch);
  sfx->createPitchNetwork(*pitchVector, results);
  scheduler::Network sfxPitchNetwork(pitchVector);
  sfxPitchNetwork.run();

  E_INFO("FreesoundExtractor: Compute aggregation");
  stats = computeAggregation(results);

  if (options.value<Real>("highlevel.compute")) {
#if HAVE_GAIA2    
    E_INFO("FreesoundExtractor: SVM models");
    _svms->input("pool").set(stats);
    _svms->output("pool").set(stats);
    _svms->compute();
#else
    E_WARNING("FreesoundExtractor: Gaia library is missing. Skipping computation of SVM models.");
#endif
  }
  E_INFO("All done");
  
  resultsStats = stats;
  resultsFrames = results;
}


Pool FreesoundExtractor::computeAggregation(Pool& pool){

  // choose which descriptors stats to output
  const char* defaultStats[] = { 
    "mean", "var", "stdev", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };

  map<string, vector<string> > exceptions;

  // do not aggregate values for features characterizing the whole audio  
  const char* noStats[] = { "last" };
  const char *noStatsSfxArray[] = {
    "der_av_after_max", "effective_duration","flatness", "logattacktime",
    "max_der_before_max", "pitch_centroid",
    "temporal_centroid","temporal_decrease" ,"temporal_kurtosis",
    "temporal_skewness","temporal_spread"};
  vector<string> noStatsSfx = arrayToVector<string>(noStatsSfxArray);

  for (int i=0; i<(int)noStatsSfx.size(); i++) {
    exceptions["sfx."+noStatsSfx[i]] = arrayToVector<string>(noStats);
  }

  // keeping this from MusicExtractor
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
  }

  standard::Algorithm* aggregator = 
    standard::AlgorithmFactory::create("PoolAggregator",
                                       "defaultStats", arrayToVector<string>(defaultStats),
                                       "exceptions", exceptions);
  Pool poolStats;
  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);
  aggregator->compute();


  // variable descriptor length counts
  poolStats.set(string("rhythm.onset_count"),
                pool.value<vector<Real> >("rhythm.onset_times").size());
  poolStats.set(string("rhythm.beats_count"),
                pool.value<vector<Real> >("rhythm.beats_position").size());
  poolStats.set(string("tonal.chords_count"),
                pool.value<vector<string> >("tonal.chords_progression").size());
    
  // hpcp_mean peak count
  vector<Real> hpcp_peak_amps, hpcp_peak_pos;
  standard::Algorithm* hpcp_peaks =
      standard::AlgorithmFactory::create("PeakDetection", "threshold",0.1);

  hpcp_peaks->input("array").set(poolStats.value<vector<Real> >("tonal.hpcp.mean"));
  hpcp_peaks->output("amplitudes").set(hpcp_peak_amps);
  hpcp_peaks->output("positions").set(hpcp_peak_pos);
  hpcp_peaks->compute();

  poolStats.set(string("tonal.hpcp_peak_count"), hpcp_peak_amps.size());

  // MusicExtractor post-processes beats loudness in different way
  /*
  // add descriptors that may be missing due to content
  const Real emptyVector[] = { 0, 0, 0, 0, 0, 0};

  int statsSize = int(sizeof(defaultStats)/sizeof(defaultStats[0]));

  if (!pool.contains<vector<Real> >("rhythm.beats_loudness")) {
    for (int i=0; i<statsSize; i++)
        poolStats.set(string("rhythm.beats_loudness.")+defaultStats[i], 0);
  }

  if (!pool.contains<vector<vector<Real> > >("rhythm.beats_loudness_band_ratio")) {
    for (int i=0; i<statsSize; i++)
      poolStats.set(string("rhythm.beats_loudness_band_ratio.")+defaultStats[i], arrayToVector<Real>(emptyVector));
  }
  */

  delete aggregator;
  delete hpcp_peaks;
  
  return poolStats;
}


void FreesoundExtractor::readMetadata(const string& audioFilename, Pool& results) {
  // Pool Connector in streaming mode currently does not support Pool sources,
  // therefore, using standard mode

  standard::Algorithm* metadata = standard::AlgorithmFactory::create("MetadataReader",
                                                                     "filename", audioFilename,
                                                                     "failOnError", true,
                                                                     "tagPoolName", "metadata.tags");
  string title, artist, album, comment, genre, tracknumber, date;
  int duration, sampleRate, bitrate, channels;

  Pool poolTags;
  metadata->output("title").set(title);
  metadata->output("artist").set(artist);
  metadata->output("album").set(album);
  metadata->output("comment").set(comment);
  metadata->output("genre").set(genre);
  metadata->output("tracknumber").set(tracknumber);
  metadata->output("date").set(date);

  metadata->output("bitrate").set(bitrate);
  metadata->output("channels").set(channels);
  metadata->output("duration").set(duration);
  metadata->output("sampleRate").set(sampleRate);

  metadata->output("tagPool").set(poolTags);

  metadata->compute();

  results.merge(poolTags);
  delete metadata;

#if defined(OS_WIN32) && !defined(OS_MINGW)
  string slash = "\\";
#else
  string slash = "/";
#endif

  string basename;
  size_t found = audioFilename.rfind(slash);
  if (found != string::npos) {
    basename = audioFilename.substr(found+1);
  } else {
    basename = audioFilename;
  }
  results.set("metadata.tags.file_name", basename);
}


void FreesoundExtractor::computeAudioMetadata(const string& audioFilename, Pool& results) {
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
  streaming::Algorithm* loader = factory.create("AudioLoader",
                                                "filename",   audioFilename,
                                                "computeMD5", true);

  loader->output("md5")             >> PC(results, "metadata.audio_properties.md5_encoded");
  loader->output("sampleRate")      >> PC(results, "metadata.audio_properties.sample_rate");
  loader->output("numberChannels")  >> PC(results, "metadata.audio_properties.number_channels");
  loader->output("bit_rate")        >> PC(results, "metadata.audio_properties.bit_rate");
  loader->output("codec")           >> PC(results, "metadata.audio_properties.codec");

  streaming::Algorithm* demuxer = factory.create("StereoDemuxer");
  streaming::Algorithm* muxer = factory.create("StereoMuxer");
  streaming::Algorithm* resampleR = factory.create("Resample");
  streaming::Algorithm* resampleL = factory.create("Resample");
  streaming::Algorithm* trimmer = factory.create("StereoTrimmer");
  streaming::Algorithm* loudness = factory.create("LoudnessEBUR128");

  Real inputSampleRate = lastTokenProduced<Real>(loader->output("sampleRate"));
  resampleR->configure("inputSampleRate", inputSampleRate,
                       "outputSampleRate", analysisSampleRate);
  resampleL->configure("inputSampleRate", inputSampleRate,
                       "outputSampleRate", analysisSampleRate);
  trimmer->configure("sampleRate", analysisSampleRate,
                     "startTime", startTime,
                     "endTime", endTime);

  // TODO implement StereoLoader algorithm instead of hardcoding this chain
  loader->output("audio")      >> demuxer->input("audio");
  demuxer->output("left")      >> resampleL->input("signal");
  demuxer->output("right")     >> resampleR->input("signal");
  resampleR->output("signal")  >> muxer->input("right");
  resampleL->output("signal")  >> muxer->input("left");
  muxer->output("audio")       >> trimmer->input("signal");
  trimmer->output("signal")    >> loudness->input("signal");
  loudness->output("integratedLoudness") >> PC(results, "lowlevel.loudness_ebu128.integrated");
  loudness->output("momentaryLoudness") >> PC(results, "lowlevel.loudness_ebu128.momentary");
  loudness->output("shortTermLoudness") >> PC(results, "lowlevel.loudness_ebu128.short_term");
  loudness->output("loudnessRange") >> PC(results, "lowlevel.loudness_ebu128.loudness_range");

  scheduler::Network network(loader);
  network.run();
  
  // set length (actually duration) of the file and length of analyzed segment
  Real length = loader->output("audio").totalProduced() / inputSampleRate;
  Real analysis_length = trimmer->output("signal").totalProduced() / analysisSampleRate;

  if (!analysis_length) {
    ostringstream msg;
    msg << "FreesoundExtractor: empty input signal (analysis startTime: " << startTime
        << ", endTime: " <<  endTime << ", input audio length: " << length << ")";
    throw EssentiaException(msg);
  }

  results.set("metadata.audio_properties.length", length);
  results.set("metadata.audio_properties.analysis.length", analysis_length);

  // This is just our best guess as to if a file is in a lossless or lossy format
  // It won't protect us against people converting from (e.g.) mp3 -> flac
  // before submitting
  const char* losslessCodecs[] = {"alac", "ape", "flac", "shorten", "tak", "truehd", "tta", "wmalossless"};
  vector<string> lossless = arrayToVector<string>(losslessCodecs);
  const string codec = results.value<string>("metadata.audio_properties.codec");
  bool isLossless = find(lossless.begin(), lossless.end(), codec) != lossless.end();
  if (!isLossless && codec.substr(0, 4) == "pcm_") {
      isLossless = true;
  }
  results.set("metadata.audio_properties.lossless", isLossless);
}


void FreesoundExtractor::computeReplayGain(const string& audioFilename, Pool& results) {

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  replayGain = 0.0;
  //int length = 0;

  while (true) {
    streaming::Algorithm* audio = factory.create("EqloudLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate,
                                      "startTime",  startTime,
                                      "endTime",    endTime,
                                      "downmix",    downmix);
    streaming::Algorithm* rgain = factory.create("ReplayGain", "applyEqloud", false);

    audio->output("audio")      >> rgain->input("signal");
    rgain->output("replayGain") >> PC(results, "metadata.audio_properties.replay_gain");

    try {
      scheduler::Network network(audio);
      network.run();
      //length = audio->output("audio").totalProduced();
      replayGain = results.value<Real>("metadata.audio_properties.replay_gain");
    }

    catch (const EssentiaException&) {
      if (downmix == "mix") {
        downmix = "left";
      }
      else {
        throw EssentiaException("File looks like a completely silent file");
      }

      try {
        results.remove("metadata.audio_properties.replay_gain");
      }
      catch (EssentiaException&) {}
      continue;
    }

    if (replayGain <= 40.0) {
      // normal replay gain value; threshold set to 20 was found too conservative
      break;
    }

    // otherwise, a very high value for replayGain: we are probably analyzing a
    // silence even though it is not a pure digital silence. except if it was
    // some electro music where someone thought it was smart to have opposite
    // left and right channels... Try with only the left channel, then.
    if (downmix == "mix") {
      downmix = "left";
      results.remove("metadata.audio_properties.replay_gain");
    }
    else {
      throw EssentiaException("File looks like a completely silent file... Aborting...");
      //exit(5);
    }
  }
}


void FreesoundExtractor::setExtractorOptions(const std::string& filename) {

  if (filename.empty()) return;

  Pool opts;
  standard::Algorithm * yaml = standard::AlgorithmFactory::create("YamlInput", "filename", filename);
  yaml->output("pool").set(opts);
  yaml->compute();
  delete yaml;
  options.merge(opts, "replace");
}

} // namespace standard
} // namespace essentia
