/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_HPCP_H
#define ESSENTIA_HPCP_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HPCP : public Algorithm {
 public:
  struct HarmonicPeak {
    HarmonicPeak(Real semitone, Real harmonicStrength = 0.0)
      : semitone(semitone), harmonicStrength(harmonicStrength) {}

    Real semitone;
    Real harmonicStrength;
  };

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _hpcp;

 public:
  HPCP() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz]");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks");
    declareOutput(_hpcp, "hpcp", "the resulting harmonic pitch class profile");
  }

  void declareParameters() {
    declareParameter("size", "the size of the output HPCP (must be a positive nonzero multiple of 12)", "[12,inf)", 12);
    declareParameter("referenceFrequency", "the reference frequency for semitone index calculation, corresponding to A3 [Hz]", "(0,inf)", 440.0);
    declareParameter("harmonics", "number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution", "[0,inf)", 0); // 8 for chord estimation
    declareParameter("bandPreset", "enables whether to use a band preset", "{true,false}", true);
    declareParameter("minFrequency", "the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)", "(0,inf)", 40.0);
    declareParameter("maxFrequency", "the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)", "(0,inf)", 5000.0);
    declareParameter("splitFrequency", "the split frequency for low and high bands, not used if bandPreset is false [Hz]", "(0,inf)", 500.0);
    declareParameter("weightType", "type of weighting function for determining frequency contribution", "{none,cosine,squaredCosine}", "squaredCosine");
    declareParameter("nonLinear", "enables whether to apply a Jordi non-linear post-processing function to the output", "{true,false}", false);
    declareParameter("windowSize", "the size, in semitones, of the window used for the weighting", "(0,12]", 1.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxShifted", "whether to shift the HPCP vector so that the maximum peak is at index 0", "{true,false}", false);
    declareParameter("normalized", "whether to normalize the HPCP vector", "{true,false}", true);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;
  static const Real precision;

 protected:
  void addContribution(Real freq, Real mag_lin, std::vector<Real>& hpcp) const;
  void addContributionWithWeight(Real freq, Real mag_lin, std::vector<Real>& hpcp, Real harmonicWeight) const;
  void addContributionWithoutWeight(Real freq, Real mag_lin, std::vector<Real>& hpcp, Real harmonicWeight) const;

  void initHarmonicContributionTable();
  int _size;
  Real _windowSize;
  Real _referenceFrequency;
  Real _nHarmonics;
  Real _minFrequency;
  Real _maxFrequency;
  Real _splitFrequency;
  bool _bandPreset;

  enum WeightType {
    NONE, COSINE, SQUARED_COSINE
  };
  WeightType _weightType;
  bool _nonLinear;
  bool _maxShifted;
  bool _normalized;

  std::vector<HarmonicPeak> _harmonicPeaks;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HPCP : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _hpcp;

 public:
  HPCP() {
    declareAlgorithm("HPCP");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_hpcp, TOKEN, "hpcp");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_HPCP_H
