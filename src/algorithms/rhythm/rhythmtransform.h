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

#ifndef ESSENTIA_RHYTHMTRANSFORM_H
#define ESSENTIA_RHYTHMTRANSFORM_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class RhythmTransform : public Algorithm {

 protected:
  Input<std::vector<std::vector<Real> > > _melBands;
  Output<std::vector<std::vector<Real> > > _rhythmTransform;

  int _rtFrameSize;
  int _rtHopSize;

  Algorithm* _w;
  Algorithm* _spec;

 public:
  RhythmTransform() {
    declareInput(_melBands, "melBands", "the energy in the melbands");
    declareOutput(_rhythmTransform, "rhythm", "consecutive frames in the rhythm domain");

    AlgorithmFactory& factory = AlgorithmFactory::instance();
    _w = factory.create("Windowing", "type", "blackmanharris62");
    _spec = factory.create("Spectrum");

  }

  ~RhythmTransform() {
    delete _w;
    delete _spec;
  }

  void declareParameters() {
    declareParameter("frameSize", "the frame size to compute the rhythm trasform", "(0,inf)", 256);
    declareParameter("hopSize", "the hop size to compute the rhythm transform", "(0,inf)", 32);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class RhythmTransform : public AlgorithmComposite {

 protected:
  Sink<std::vector<Real> > _melbands;
  // it has to be a TNT::Array cause Pool doesn't support vector<vector<type> >
  Source<TNT::Array2D<Real> > _rhythmTransform;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _rhythmAlgo;

 public:
  RhythmTransform();
  ~RhythmTransform();

  void declareParameters() {
    declareParameter("frameSize", "the frame size to compute the rhythm trasform", "(0,inf)", 256);
    declareParameter("hopSize", "the hop size to compute the rhythm transform", "(0,inf)", 32);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_RHYTHMTRANSFORM_H
