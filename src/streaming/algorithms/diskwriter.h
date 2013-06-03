/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
