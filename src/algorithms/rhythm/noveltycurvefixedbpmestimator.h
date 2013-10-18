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

#ifndef ESSENTIA_NOVELTY_CURVE_FIXED_BPM_ESTIMATOR_H
#define ESSENTIA_NOVELTY_CURVE_FIXED_BPM_ESTIMATOR_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {
class NoveltyCurveFixedBpmEstimator : public Algorithm {

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
    NoveltyCurveFixedBpmEstimator() {
    declareInput(_novelty, "novelty", "the novelty curve of the audio signal");
    declareOutput(_bpmPositions, "bpms", "the bpm candidates sorted by magnitude");
    declareOutput(_bpmAmplitudes, "amplitudes", "the magnitude of each bpm candidate");
    _autocor = AlgorithmFactory::create("AutoCorrelation",
                                        "normalization", "unbiased");
    }
    ~NoveltyCurveFixedBpmEstimator(){
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

#endif // ESSENTIA_NOVELTY_CURVE_FIXED_BPM_ESTIMATOR_H
