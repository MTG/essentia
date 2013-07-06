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

#ifndef FADEDETECTION_MEAN_H
#define FADEDETECTION_MEAN_H

#include "algorithmfactory.h"
#include "tnt/tnt.h"

namespace essentia {
namespace standard {

class FadeDetection : public Algorithm {

 private:
  Input<std::vector<Real> > _rms;
  Output<TNT::Array2D<Real> > _fade_in;
  Output<TNT::Array2D<Real> > _fade_out;

  Real _frameRate;
  Real _cutoffHigh;
  Real _cutoffLow;
  Real _minLength;

 public:
  FadeDetection() {
    declareInput(_rms, "rms", "rms values array");
    declareOutput(_fade_in, "fadeIn", "2D-array containing start/stop timestamps corresponding to fade-ins [s] (ordered chronologically)");
    declareOutput(_fade_out, "fadeOut", "2D-array containing start/stop timestamps corresponding to fade-outs [s] (ordered chronologically)");
  }

  void declareParameters() {
    declareParameter("frameRate", "the rate of frames used in calculation of the RMS [frames/s]", "(0,inf)", 4.0);
    declareParameter("cutoffHigh", "fraction of the average RMS to define the maximum threshold", "(0,1]", 0.85);
    declareParameter("cutoffLow", "fraction of the average RMS to define the minimum threshold", "[0,1)", 0.20);
    declareParameter("minLength", "the minimum length to consider a fade-in/out [s]", "(0,inf)", 3.0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {

class FadeDetection : public Algorithm {

 protected:
  Sink<Real> _rms;

  Source<TNT::Array2D<Real> > _fade_in;
  Source<TNT::Array2D<Real> > _fade_out;

  std::vector<Real> _accu;

  standard::Algorithm* _fadeAlgo;

 public:

  FadeDetection() {
    declareInput(_rms, 1, "rms", "rms values array");
    declareOutput(_fade_in, 0, "fadeIn", "2D-array containing start/stop timestamps corresponding to fade-ins [s] (ordered chronologically)");
    declareOutput(_fade_out, 0, "fadeOut", "2D-array containing start/stop timestamps corresponding to fade-outs [s] (ordered chronologically)");
    _fadeAlgo = standard::AlgorithmFactory::create("FadeDetection");
  }

  ~FadeDetection() {
    delete _fadeAlgo;
  }

  void declareParameters() {
    declareParameter("frameRate", "the rate of frames used in calculation of the RMS [frames/s]", "(0,inf)", 4.0);
    declareParameter("cutoffHigh", "fraction of the average RMS to define the maximum threshold", "(0,1]", 0.85);
    declareParameter("cutoffLow", "fraction of the average RMS to define the minimum threshold", "[0,1)", 0.20);
    declareParameter("minLength", "the minimum length to consider a fade-in/out [s]", "(0,inf)", 3.0);
  }

  void configure();
  AlgorithmStatus process();
  void reset();
};

} // namespace streaming
} // namespace essentia


#endif // FADEDETECTION_MEAN_H
