/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TUNINGFREQUENCY_H
#define ESSENTIA_TUNINGFREQUENCY_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class TuningFrequency : public Algorithm {

 private:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<Real> _tuningCents;
  Output<Real> _tuningFrequency;

 public:
  TuningFrequency() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz]");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks");
    declareOutput(_tuningFrequency, "tuningFrequency", "the tuning frequency [Hz]");
    std::ostringstream tuningCentsDescription;
    tuningCentsDescription << "the deviation from 440 Hz (between " << wrappingBoundary << " to " << (wrappingBoundary + 100) << " cents)";
    declareOutput(_tuningCents, "tuningCents", tuningCentsDescription.str());
  }

  void declareParameters() {
    declareParameter("resolution", "resolution in cents (logarithmic scale, 100 cents = 1 semitone) for tuning frequency determination", "(0,inf)", 1.0);
  }

  void compute();
  void configure();
  void reset();

  static const char* name;
  static const char* description;

  // -35: estimated by looking at the distribution histogram of the tuning
  // frequency values and got the lowest ones, ie: the less likely to have songs
  // which have both higher and lower TF than this one.
  // There is no silver bullet here, but this tries to minimize the error we
  // might make
  static const Real wrappingBoundary;

 protected:
  Real _resolution;
  std::vector<Real> _histogram;
  std::vector<Real> _globalHistogram;

  Real currentTuningCents() const;
  Real tuningFrequencyFromCents(Real cents) const;
  void updateOutputs();
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TuningFrequency : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<Real> _tuningCents;
  Source<Real> _tuningFrequency;

 public:
  TuningFrequency() {
    declareAlgorithm("TuningFrequency");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_tuningFrequency, TOKEN, "tuningFrequency");
    declareOutput(_tuningCents, TOKEN, "tuningCents");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TUNINGFREQUENCY_H
