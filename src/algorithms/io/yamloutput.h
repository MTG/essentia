/*
 * Copyright (C) 2006-2009 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
