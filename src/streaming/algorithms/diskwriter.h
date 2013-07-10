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

#ifndef ESSENTIA_DISKWRITER_H
#define ESSENTIA_DISKWRITER_H

#include <fstream>
#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

template <class T>
class DiskWriter : public Algorithm {
 protected:
  Sink<T> _data;

  std::string _filename;
  std::ostream *_out;

 public:
  DiskWriter(const std::string& filename) : Algorithm() {

    declareInput(_data, 1, "data", "the data to write to disk");
    _name = "DiskWriter";

    if (filename != "-")
      _out = new std::ofstream(_filename.c_str());
    else
      _out = &std::cout;
  }

  ~DiskWriter() {
    if (_out != &std::cout) {
      delete _out;
    }
  }

  void declareParameters() {}

  AlgorithmStatus process() {
    EXEC_DEBUG("process()");

    AlgorithmStatus status = acquireData();
    if (status != OK) return status;

    EXEC_DEBUG("data acquired");

    *_out << _data.firstToken() << '\n';

    EXEC_DEBUG("produced data");

    EXEC_DEBUG("releasing");
    releaseData();
    EXEC_DEBUG("released");

    return OK;
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DISKWRITER_H
