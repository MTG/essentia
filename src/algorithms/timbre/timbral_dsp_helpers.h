#ifndef ESSENTIA_TIMBRAL_DSP_HELPERS_H
#define ESSENTIA_TIMBRAL_DSP_HELPERS_H

#include "types.h"

#include <vector>
#include <string>
#include <tuple>
#include <cstddef>

namespace essentia {

class TimbralDspHelpers {
public:
    // Loudness normalization (ITU-R BS.1770-4 / EBU R128)
    static std::vector<Real> normalizeLoudnessEbuR128(const std::vector<Real>& audio,
                                                      Real sampleRate = 44100.0f,
                                                      Real targetLufs = -24.0f);

    static Real measureIntegratedLoudnessEbuR128(const std::vector<Real>& audio,
                                                Real sampleRate = 44100.0f);

    // Cascaded Butterworth IIR filters
    static std::vector<Real> cascadeButterworthHighPass(const std::vector<Real>& signal,
                                                        Real cutoffFreq,
                                                        Real sampleRate = 44100.0f,
                                                        int orderStages = 3,
                                                        int biquadOrder = 1);

    static std::vector<Real> cascadeButterworthLowPass(const std::vector<Real>& signal,
                                                       Real cutoffFreq,
                                                       Real sampleRate = 44100.0f,
                                                       int orderStages = 3,
                                                       int biquadOrder = 1);

    // Windows & spectrum
    static std::vector<Real> hammingWindow(int size);
    static std::vector<Real> hannWindow(int size, bool periodic = true);
    static std::vector<Real> powerSpectrum(const std::vector<Real>& frame, int nFft);
    static std::vector<Real> magnitudeSpectrum(const std::vector<Real>& frame, int nFft);
    static Real spectralCentroid(const std::vector<Real>& powerSpec, Real sampleRate);

    // Peak picking and roughness
    static std::tuple<std::vector<int>, std::vector<Real>, std::vector<Real>>
    detectPeaksMIR(const std::vector<Real>& array,
                   const std::vector<Real>& freqs,
                   Real cthr = 0.01f,
                   const std::vector<Real>& unprocessed = {});

    static Real computeFrameRoughness(const std::vector<Real>& peakFreqs,
                                      const std::vector<Real>& peakLevels);

    static Real computeRoughness(const std::vector<Real>& audio,
                                 Real sampleRate = 44100.0f,
                                 bool clipOutput = false);

    static Real r2LinearFit(const std::vector<Real>& x, const std::vector<Real>& y);

    // HPSS, bandwidth trajectory & attack analysis
    static Real calculateHpssRatio(const std::vector<Real>& audio,
                                  Real sampleRate = 44100.0f);

    static std::vector<Real> calculateBandwidthArray(const std::vector<Real>& audio,
                                                     Real sampleRate = 44100.0f,
                                                     int nperseg = 4096,
                                                     int hop = 128,
                                                     Real thresholdDb = -75.0f);

    static std::tuple<Real, Real, int> calculateAttackTime(const std::vector<Real>& envelopeSamples,
                                                           Real sampleRate,
                                                           Real maxAttackTime = 0.1f,
                                                           const std::string& calculationType = "min_effort");

    static Real calculateAttackCentroid(const std::vector<Real>& audioSeg,
                                        Real sampleRate = 44100.0f);
};

} // namespace essentia

#endif // ESSENTIA_TIMBRAL_DSP_HELPERS_H
