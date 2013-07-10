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

#ifndef ESSENTIA_YAML_OUTPUT_H
#define ESSENTIA_YAML_OUTPUT_H

#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class YamlOutput : public Algorithm {

 protected:
  Input<Pool> _pool;
  std::string _filename;
  bool _doubleCheck;
  bool _outputJSON;

  void outputToStream(std::ostream* out);

 public:

  YamlOutput() {
    declareInput(_pool, "pool", "Pool to serialize into a YAML formatted file");
  }

  void declareParameters() {
    declareParameter("filename", "output filename (use '-' to emit to stdout)", "", "-");
    declareParameter("doubleCheck", "whether to double-check if the file has been correctly written to the disk", "", false);
    declareParameter("format", "whether to output data in JSON or YAML format", "{json,yaml}", "yaml");
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#endif // ESSENTIA_EXTRACTOR_YAML_OUTPUT_H
