/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_INHARMONICITY_H
#define ESSENTIA_INHARMONICITY_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Inharmonicity : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<Real> _inharmonicity;

 public:
  Inharmonicity() {
    declareInput(_frequencies, "frequencies", "the frequencies of the harmonic peaks [Hz] (in ascending order)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the harmonic peaks (in frequency ascending order");
    declareOutput(_inharmonicity, "inharmonicity", "the inharmonicity of the audio signal");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Inharmonicity : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<Real> _inharmonicity;

 public:
  Inharmonicity() {
    declareAlgorithm("Inharmonicity");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_inharmonicity, TOKEN, "inharmonicity");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_INHARMONICITY_H
