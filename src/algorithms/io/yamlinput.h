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

#ifndef ESSENTIA_YAML_INPUT_H
#define ESSENTIA_YAML_INPUT_H

#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class YamlInput : public Algorithm {

 protected:
  Output<Pool> _pool;
  std::string _filename;

 public:
  YamlInput() {
    declareOutput(_pool, "pool", "Pool of deserialized values");
  }

  void declareParameters() {
    declareParameter("filename", "Input filename (must be in YAML format)", "", Parameter::STRING);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_EXTRACTOR_YAML_INPUT_H
