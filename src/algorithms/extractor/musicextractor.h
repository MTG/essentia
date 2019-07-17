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

#ifndef MUSIC_EXTRACTOR_H
#define MUSIC_EXTRACTOR_H

#include "pool.h"
#include "algorithm.h"

#include "extractor_music/MusicLowlevelDescriptors.h"
#include "extractor_music/MusicRhythmDescriptors.h"
#include "extractor_music/MusicTonalDescriptors.h"
#include "extractor_music/extractor_version.h"

namespace essentia {
namespace standard {

class MusicExtractor : public Algorithm {
 protected:
  Input<std::string> _audiofile;
  Output<Pool> _resultsStats;
  Output<Pool> _resultsFrames;  

  Real analysisSampleRate;
  Real startTime;
  Real endTime;
  bool requireMbid;

  int lowlevelFrameSize;
  int lowlevelHopSize;
  int lowlevelZeroPadding;
  std::string lowlevelSilentFrames;
  std::string lowlevelWindowType;

  int tonalFrameSize;
  int tonalHopSize;
  int tonalZeroPadding;
  std::string tonalSilentFrames;
  std::string tonalWindowType;

  int loudnessFrameSize;
  int loudnessHopSize;
  //std::string loudnessSilentFrames;
  //std::string loudnessWindowType;

  std::string rhythmMethod;
  int rhythmMinTempo;
  int rhythmMaxTempo;

  std::vector<std::string> lowlevelStats;
  std::vector<std::string> tonalStats;
  std::vector<std::string> rhythmStats;
  std::vector<std::string> mfccStats;
  std::vector<std::string> gfccStats;

#if HAVE_GAIA2 
  std::vector<std::string> svmModels;
#endif

  Real replayGain;
  std::string downmix;
  standard::Algorithm* _svms;

  void setExtractorOptions(const std::string& filename);
  void setExtractorDefaultOptions();
  void mergeValues(Pool &pool);
  void readMetadata(const std::string& audioFilename, Pool& results);
  void computeAudioMetadata(const std::string& audioFilename, Pool& results);
  void computeReplayGain(const std::string& audioFilename, Pool& results);

  Pool computeAggregation(Pool& pool);

 public:

  MusicExtractor();
  ~MusicExtractor();

  void declareParameters() {
    declareParameter("profile", "profile filename. If specified, default parameter values are overwritten by values in the profile yaml file. If not specified (empty string), use values configured by user like in other normal algorithms", "", Parameter::STRING);
    
    declareParameter("analysisSampleRate", "the analysis sampling rate of the audio signal [Hz]", "(0,inf)", 44100.0);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
    declareParameter("requireMbid", "ignore audio files without musicbrainz recording id tag (throw exception)", "{true,false}", false);
    // requireMbid option is very specific for AcousticBrainz extractor
    // however, we'll keep it here for now...
  
    declareParameter("lowlevelFrameSize", "the frame size for computing low-level features", "(0,inf)", 2048);
    declareParameter("lowlevelHopSize", "the hop size for computing low-level features", "(0,inf)", 1024);
    declareParameter("lowlevelZeroPadding", "zero padding factor for computing low-level features", "[0,inf)", 0);
    declareParameter("lowlevelSilentFrames", "whether to [keep/drop/add noise to] silent frames for computing low-level features", "{drop,keep,noise}", "noise");
    declareParameter("lowlevelWindowType", "the window type for computing low-level features", "{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "blackmanharris62");

    declareParameter("tonalFrameSize", "the frame size for computing tonal features", "(0,inf)", 4096);
    declareParameter("tonalHopSize", "the hop size for computing tonal features", "(0,inf)", 2048);
    declareParameter("tonalZeroPadding", "zero padding factor for computing tonal features", "[0,inf)", 0);
    declareParameter("tonalSilentFrames", "whether to [keep/drop/add noise to] silent frames for computing tonal features", "{drop,keep,noise}", "noise");
    declareParameter("tonalWindowType", "the window type for computing tonal features", "{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "blackmanharris62");

    // TODO average_loudness is redundant? we compare with replaygain and ebu r128
    declareParameter("loudnessFrameSize", "the frame size for computing average loudness", "(0,inf)", 88200);
    declareParameter("loudnessHopSize", "the hop size for computing average loudness", "(0,inf)", 44100);
    //declareParameter("loudnessWindowType", "the window type for computing average loudness", "{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    //declareParameter("loudnessSilentFrames", "whether to [keep/drop/add noise to] silent frames for computing average loudness", "{drop,keep,noise}", "noise");

    declareParameter("rhythmMethod", "the method used for beat tracking", "{multifeature,degara}", "degara");
    declareParameter("rhythmMinTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
    declareParameter("rhythmMaxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
  
    const char* statsArray[] = { "mean", "var", "stdev", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };
    const char* cepstrumStatsArray[] = { "mean", "cov", "icov" };
    vector<string> stats = arrayToVector<string>(statsArray);
    vector<string> cepstrumStats = arrayToVector<string>(cepstrumStatsArray);

    declareParameter("lowlevelStats", "the statistics to compute for low-level features", "", stats);
    declareParameter("tonalStats", "the statistics to compute for tonal features", "", stats);
    declareParameter("rhythmStats", "the statistics to compute for rhythm features", "", stats);
    
    declareParameter("mfccStats", "the statistics to compute for MFCC features", "", cepstrumStats);
    declareParameter("gfccStats", "the statistics to compute for GFCC features", "", cepstrumStats);

#if HAVE_GAIA2 
    declareParameter("highlevel", "list of high-level classifier models (gaia2 history filenames) to apply using extracted features. Skip classification if not specified (empty list)", "", Parameter::VECTOR_STRING);
#endif
  }

  Pool options;

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif
