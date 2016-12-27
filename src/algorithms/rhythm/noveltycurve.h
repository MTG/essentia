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

#ifndef ESSENTIA_NOVELTYCURVE_H
#define ESSENTIA_NOVELTYCURVE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class NoveltyCurve : public Algorithm {

 protected:
  Input<std::vector<std::vector<Real> > > _frequencyBands;
  Output<std::vector<Real> > _novelty;

  enum WeightType {
    FLAT, TRIANGLE, INVERSE_TRIANGLE, PARABOLA, INVERSE_PARABOLA,
    LINEAR, QUADRATIC, INVERSE_QUADRATIC, SUPPLIED, HYBRID
  };

  Real _frameRate;
  WeightType _type;
  bool _normalize;

  std::vector<Real> weightCurve(int size, WeightType type);
  std::vector<Real> noveltyFunction(const std::vector<Real>& spec, Real C, int meanSize);

 public:

  NoveltyCurve() {
    declareInput(_frequencyBands, "frequencyBands", "the frequency bands");
    declareOutput(_novelty, "novelty", "the novelty curve as a single vector");
  }

  ~NoveltyCurve() {}

  void declareParameters() {
    declareParameter("frameRate", "the sampling rate of the input audio", "[1,inf)", 44100./128.);
    declareParameter("weightCurveType", "the type of weighting to be used for the bands novelty","{flat,triangle,inverse_triangle,parabola,inverse_parabola,linear,quadratic,inverse_quadratic,hybrid,supplied}",
                     "hybrid");
    declareParameter("weightCurve", "vector containing the weights for each frequency band. Only if weightCurveType==supplied", "", std::vector<Real>(0));
    declareParameter("normalize", "whether to normalize each band's energy", "{true,false}", false);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class NoveltyCurve : public AlgorithmComposite {

 protected:
  SinkProxy<std::vector<Real> > _frequencyBands;
  Source<Real> _novelty;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm * _noveltyCurve;

 public:
  NoveltyCurve();
  ~NoveltyCurve();

  void declareParameters() {
    declareParameter("frameRate", "the sampling rate of the input audio", "[1,inf)", 44100./128.);
    declareParameter("weightCurveType", "the type of weighting to be used for the bands novelty","{flat,triangle,inverse_triangle,parabola,inverse_parabola,linear,quadratic,inverse_quadratic,supplied}",
                     "inverse_quadratic");
    declareParameter("weightCurve", "vector containing the weights for each frequency band. Only if weightCurveType==supplied", "", std::vector<Real>(0));
    declareParameter("normalize", "whether to normalize each band's energy", "{true,false}", false);
  }

  void configure() {
    _noveltyCurve->configure(INHERIT("frameRate"),
                               INHERIT("weightCurveType"),
                               INHERIT("weightCurve"),
                               INHERIT("normalize"));
  }

  void declareProcessOrder() {
    declareProcessStep(SingleShot(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_NOVELTYCURVE_H
