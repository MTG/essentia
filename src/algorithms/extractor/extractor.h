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

#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "algorithm.h"
#include "pool.h"
#include "vectorinput.h"

namespace essentia {
namespace standard {

class Extractor : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Pool> _pool;

  Real _sampleRate;
  std::string _ns, _llspace, _sfxspace, _rhythmspace, _tonalspace; // namespaces
  bool _lowLevel, _tuning, _dynamics, _rhythm, _midLevel, _highLevel, _relativeIoi;
  int _lowLevelFrameSize, _lowLevelHopSize, _tonalFrameSize,
    _tonalHopSize, _dynamicsFrameSize, _dynamicsHopSize;

  void connectLowLevel(streaming::VectorInput<Real>*, Pool&);
  void connectTuning(streaming::VectorInput<Real>*, Pool&);
  void connectDynamics(streaming::VectorInput<Real>*, Pool&);
  void connectRhythm(streaming::VectorInput<Real>*, Pool&);
  void computeMidLevel(const std::vector<Real>&, Pool&);
  void computeHighLevel(Pool&);
  void computeRelativeIoi(Pool&);
  void postProcessOnsetRate(streaming::VectorInput<Real>*, Pool&);
  void levelAverage(Pool&);
  void tuningSystemFeatures(Pool&);
  void sfxPitch(Pool&);
  Real squeezeRange(Real&, Real&, Real&);


 public:
  Extractor() {
    declareInput(_signal, "audio", "the input audio signal");
    declareOutput(_pool, "pool", "the pool where to store the results");
  }
  ~Extractor() {}

  void declareParameters() {
    declareParameter("lowLevelFrameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("lowLevelHopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("tonalFrameSize", "the frame size for low level tonal features", "(0,inf)", 4096 );
    declareParameter("tonalHopSize", "the hop size for low level tonal features", "(0,inf)", 2048);
    declareParameter("dynamicsFrameSize", "the frame size for level dynamics", "(0,inf)", 88200);
    declareParameter("dynamicsHopSize", "the hop size for level dynamics", "(0,inf)", 44100);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
    declareParameter("namespace", "the main namespace under which to store the results", "", "");
    declareParameter("lowLevel", "compute low level features", "{true,false}", true);
    declareParameter("tuning", "compute tuning frequency", "{true,false}", true);
    declareParameter("dynamics", "compute dynamics' features", "{true,false}", true);
    declareParameter("rhythm", "compute rhythm features", "{true,false}", true);
    declareParameter("midLevel", "compute mid level features", "{true,false}", true);
    declareParameter("highLevel", "compute high level features", "{true,false}", true);
    // needed by freesound
    declareParameter("relativeIoi", "compute relative inter onset intervals", "{true,false}", false);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif
