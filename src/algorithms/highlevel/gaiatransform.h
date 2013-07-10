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
