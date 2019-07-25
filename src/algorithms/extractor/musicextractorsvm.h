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

#ifndef MUSIC_EXTRACTOR_SVM_H
#define MUSIC_EXTRACTOR_SVM_H

#include "pool.h"
#include "algorithm.h"
#include "extractor_music/extractor_version.h"

namespace essentia {
namespace standard {

class MusicExtractorSVM : public Algorithm {
 protected:
  Input<Pool> _inputPool;
  Output<Pool> _outputPool;

  std::vector<standard::Algorithm*> _svms;


 public:

  MusicExtractorSVM();
  ~MusicExtractorSVM();

  void declareParameters() {
    declareParameter("svms", "list of svm models (gaia2 history) filenames.", "", Parameter::VECTOR_STRING);
  }

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
