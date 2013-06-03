/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FIXED_BPM_ESTIMATOR_H
#define ESSENTIA_FIXED_BPM_ESTIMATOR_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {
class FixedBpmEstimator : public Algorithm {

  protected:
    Input<std::vector<Real> > _novelty;
    Output<std::vector<Real> > _bpmPositions;
    Output<std::vector<Real> > _bpmAmplitudes;
    // TODO: output a feature that tells whether the signal can be considered
    // to have constant tempo or not:
    //Output<int> _constantTempo;[0, 1, 0.5] or maybe a boolean
    // this could well be done by analyzing the shape of the resulting
    // bpmPositions, if there are too many different bpms would mean that the
    // tempo is fuzzy...

    Real _sampleRate;
    Real _minBpm;
    Real _maxBpm;
    Real _bpmTolerance;
    int _hopSize;

    // inner algos
    Algorithm* _autocor;

    Real computeTatum(const std::vector<Real>& peaks);
    Real mainPeaksMean(const std::vector<Real>& positions,
                       const std::vector<Real>& amplitudes, int size);
    void inplaceMergeBpms(std::vector<Real>& bpms, std::vector<Real>& amplitudes);
    void histogramPeaks(const std::vector<Real>& bpms,
                        std::vector<Real>& positions, std::vector<Real>& amplitudes);

  public:
    FixedBpmEstimator() {
    declareInput(_novelty, "novelty", "the novelty curve of the audio signal");
    declareOutput(_bpmPositions, "bpms", "the bpm candidates sorted by magnitude");
    declareOutput(_bpmAmplitudes, "amplitudes", "the magnitude of each bpm candidate");
    _autocor = AlgorithmFactory::create("AutoCorrelation",
                                        "normalization", "unbiased");
    }
    ~FixedBpmEstimator(){
      delete _autocor;
    }

    void declareParameters() {
      declareParameter("sampleRate", "the sampling rate original audio signal [Hz]", "[1,inf)", 44100.);
      declareParameter("hopSize", "the hopSize used to computeh the novelty curve from the original signal", "", 512);
      declareParameter("minBpm", "the minimum bpm to look for", "(0,inf)", 30.0);
      declareParameter("maxBpm", "the maximum bpm to look for", "(0,inf)", 560.0);
      declareParameter("tolerance", "tolerance (in percentage) for considering bpms to be equal", "(0,100]", 3.0);
    }

    void configure();
    void compute();
    void reset() {}

    static const char* name;
    static const char* version;
    static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_FIXED_BPM_ESTIMATOR_H
