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
  Real indent;

  Real replayGain;
  std::string downmix;
  std::vector<standard::Algorithm*> svms;

  //Pool options;

  void setExtractorOptions(const std::string& filename);
  void setExtractorDefaultOptions();
  void mergeValues(Pool &pool);
  void readMetadata(const std::string& audioFilename, Pool& results);
  void computeMetadata(const std::string& audioFilename, Pool& results);
  void computeLoudnessEBUR128(const std::string& audioFilename, Pool& results);
  void computeReplayGain(const std::string& audioFilename, Pool& results);
  void computeSVMDescriptors(Pool& pool);
  void loadSVMModels();
  void outputToFile(Pool& pool, const std::string& outputFilename);

  Pool computeAggregation(Pool& pool);

 public:

  MusicExtractor();
  ~MusicExtractor();

  void declareParameters() {
    declareParameter("profile", "profile filename. If specified, default parameter values are overwritten by values in the profile yaml file. If not specified (empty string), use values configured by user like in other normal algorithms", "", Parameter::STRING);
    // TODO implement parameters directly in addition to profile. If there's no profile then read normal parameter values. 
  }

  Pool options;

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
  static const char* version;  
};

} // namespace standard
} // namespace essentia

#endif
