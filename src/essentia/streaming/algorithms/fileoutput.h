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

#ifndef ESSENTIA_FILEOUTPUT_H
#define ESSENTIA_FILEOUTPUT_H

#include <fstream>
#include "../streamingalgorithm.h"
#include "../../streamutil.h"

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
