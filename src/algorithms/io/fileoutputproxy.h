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

#ifndef ESSENTIA_FILEOUTOUT_PROXY_H
#define ESSENTIA_FILEOUTOUT_PROXY_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class FileOutputProxy : public Algorithm {
 protected:
  Algorithm* _file;
  bool _configured;

 public:
  FileOutputProxy() : Algorithm(), _file(0), _configured(false) {
    declareParameters();
  }

  ~FileOutputProxy() {
    delete _file;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the output file", "", "out.txt");
    declareParameter("mode", "output mode", "{text,binary}", "text");
  }

  AlgorithmStatus process() {
    if (!_configured) {
      throw EssentiaException("FileOutputProxy: trying to run without proper configuration.");
    }
    return _file->process();
  }

  void setFileStorage(Algorithm* fs) {
    _file = fs;
    if (_file) _configured = true;
  }

  void reset() {
    Algorithm::reset();
    _file->reset();
  }

  static const char* name;
  static const char* description;

};

void connect(SourceBase& source, FileOutputProxy& file);
void connect(SourceBase& source, Algorithm& file);

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_FILEOUTOUT_PROXY_H
