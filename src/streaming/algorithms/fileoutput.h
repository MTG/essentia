/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FILEOUTPUT_H
#define ESSENTIA_FILEOUTPUT_H

#include <fstream>
#include "streamingalgorithm.h"
#include "streamutil.h"

namespace essentia {
namespace streaming {

template <typename TokenType, typename StorageType = TokenType>
class FileOutput : public Algorithm {
 protected:
  Sink<TokenType> _data;
  std::ostream* _stream;
  std::string _filename;
  bool _binary;

 public:
  FileOutput() : Algorithm(), _stream(NULL) {
    setName("FileOutput");
    declareInput(_data, 1, "data", "the incoming data to be stored in the output file");

    declareParameters();
  }

  ~FileOutput() {
    if (_stream != &std::cout) delete _stream;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the output file", "", "out.txt");
    declareParameter("mode", "output mode", "{text,binary}", "text");
  }

  void configure() {
    if (!parameter("filename").isConfigured()) {
      throw EssentiaException("FileOutput: please provide the 'filename' parameter");
    }

    _filename = parameter("filename").toString();

    if (_filename == "") {
      throw EssentiaException("FileOutput: empty filenames are not allowed.");
    }

    _binary = (parameter("mode").toString() == "binary");
  }

  void createOutputStream() {
    if (_filename == "-") {
      _stream = &std::cout;
    }
    else {
      _stream = _binary ? new std::ofstream(_filename.c_str(), std::ofstream::binary)
                        : new std::ofstream(_filename.c_str());

      if (_stream->fail()) {
        throw EssentiaException("FileOutput: Could not open file for writing: ", _filename);
      }
    }
  }

  AlgorithmStatus process() {
    if (!_stream) {
      createOutputStream();
    }

    EXEC_DEBUG("process()");

    if (!_data.acquire(1)) return NO_INPUT;

    write(_data.firstToken());

    _data.release(1);

    return OK;
  }

  void write(const TokenType& value) {
    if (!_stream) throw EssentiaException("FileOutput: not configured properly");
    if (_binary) {
      _stream->write((const char*) &value, sizeof(TokenType));
    }
    else {
      *_stream << value << "\n";
    }
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FILEOUTPUT_H
