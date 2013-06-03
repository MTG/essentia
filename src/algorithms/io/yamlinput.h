/*
 * Copyright (C) 2006-2009 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
