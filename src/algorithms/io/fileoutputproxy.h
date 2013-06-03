/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
