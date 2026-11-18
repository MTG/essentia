#ifndef ESSENTIA_TIMBRAL_ONSET_ENVELOPE_H
#define ESSENTIA_TIMBRAL_ONSET_ENVELOPE_H

#include "types.h"

#include <vector>

namespace essentia {

class TimbralOnsetEnvelope {
public:
    struct MelFilterbank {
        int nMels = 128;
        int nFft = 2048;
        std::vector<Real> weights;
    };

    static std::vector<Real> sampleAndHoldEnvelope(
        const std::vector<Real>& audio,
        Real sampleRate = 44100.0f,
        Real decayTime = 0.1f,
        Real holdTime = 0.01f);

    static MelFilterbank initMelFilterbank(
        Real sampleRate = 44100.0f,
        int nFft = 2048,
        int nMels = 128);

    static const MelFilterbank& defaultMelFilterbank();

    static std::vector<Real> onsetStrength(
        const std::vector<Real>& audio,
        Real sampleRate = 44100.0f,
        int nFft = 2048,
        int hop = 512);

    static std::vector<int> peakPick(
        const std::vector<Real>& x,
        int preMax,
        int postMax,
        int preAvg,
        int postAvg,
        Real delta,
        int wait);

    static std::vector<int> onsetBacktrack(
        const std::vector<int>& events,
        const std::vector<Real>& energy);

    static std::vector<int> onsetDetect(
        const std::vector<Real>& audio,
        Real sampleRate = 44100.0f,
        int hop = 512);

    static int returnLoop(
        int onsetLoc,
        const std::vector<Real>& envelope,
        int functionTimeThresh,
        Real histThreshold,
        int histTimeSamples,
        int nperseg = 512);

    static std::vector<int> calculateOnsets(
        const std::vector<Real>& audio,
        const std::vector<Real>& envelope,
        Real sampleRate = 44100.0f,
        Real lookBackTime = 20.0f,
        Real hysteresisTime = 300.0f,
        Real hysteresisPercent = 10.0f,
        Real onsetInNoiseThreshold = 10.0f,
        Real minOnsetSeparationMs = 100.0f,
        int nperseg = 512);
};

} // namespace essentia

#endif // ESSENTIA_TIMBRAL_ONSET_ENVELOPE_H
