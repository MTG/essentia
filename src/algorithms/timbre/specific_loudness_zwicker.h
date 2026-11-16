#ifndef ESSENTIA_SPECIFIC_LOUDNESS_ZWICKER_H
#define ESSENTIA_SPECIFIC_LOUDNESS_ZWICKER_H

#include "types.h"

#include <vector>

namespace essentia {

// ISO 532B Zwicker specific loudness and psychoacoustic metrics
class SpecificLoudnessZwicker {
public:
    SpecificLoudnessZwicker() = default;
    ~SpecificLoudnessZwicker() = default;

    void compute(const std::vector<Real>& audioFrame,
                 Real& totalLoudness,
                 std::vector<Real>& specificLoudness,
                 Real pRef = 100.0f,
                 int mode = 0);

    static void computeThirdOctaveLevels(const std::vector<Real>& audioFrame,
                                         std::vector<Real>& bandLevels,
                                         Real pRef = 100.0f);

    static Real computeSharpnessFastl(const std::vector<Real>& specificLoudness);

    static void computeBoominessMetrics(const std::vector<Real>& specificLoudness,
                                        Real& frameBoomingIndex,
                                        Real& lowBarkEnergy);

    static void computeWarmthRegionLevels(const std::vector<Real>& specificLoudness,
                                          Real& warmthRegionLevel,
                                          Real& highFreqLevel);

    static Real computeHardnessMidbandLevel(const std::vector<Real>& specificLoudness);
};

} // namespace essentia

#endif // ESSENTIA_SPECIFIC_LOUDNESS_ZWICKER_H
