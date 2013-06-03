/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
  SinkProxy<std::vector<Real> > _melbands;
  Source<std::vector<std::vector<Real> > > _rhythmTransform;

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
