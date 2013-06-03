/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_GAIATRANSFORM_H
#define ESSENTIA_GAIATRANSFORM_H

#include <gaia2/transformation.h>
#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class GaiaTransform : public Algorithm {

 protected:
  Input<Pool> _inputPool;
  Output<Pool> _outputPool;

  // the history of the applied transformations in Gaia
  gaia2::TransfoChain _history;

  bool _configured;

 public:
  GaiaTransform() : _configured(false) {
    declareInput(_inputPool, "pool", "aggregated pool of extracted values");
    declareOutput(_outputPool, "pool", "pool resulting from the transformation of the gaia point");

    // call it from here, the place where it's gonna called less often
    gaia2::init();
  }

  ~GaiaTransform();

  void declareParameters() {
    declareParameter("history", "gaia2 history filename", "", Parameter::STRING);
  }

  void compute();
  void configure();
  void reset() {}

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_GAIATRANSFORM_H
