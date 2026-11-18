#include "audiocommonstimbralmodels.h"
#include "specific_loudness_zwicker.h"
#include "timbral_onset_envelope.h"
#include "timbral_dsp_helpers.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cassert>

using namespace std;
using namespace essentia;
using namespace standard;

const char* AudioCommonsTimbralModels::name = "AudioCommonsTimbralModels";
const char* AudioCommonsTimbralModels::category = "Timbre";
const char* AudioCommonsTimbralModels::description = DOC("Computes the 7 Audio Commons timbral models (brightness, depth, hardness, roughness, sharpness, warmth, and boominess) on audio signals.");

void AudioCommonsTimbralModels::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _computeBrightness = parameter("computeBrightness").toBool();
  _computeDepth = parameter("computeDepth").toBool();
  _computeHardness = parameter("computeHardness").toBool();
  _computeRoughness = parameter("computeRoughness").toBool();
  _computeSharpness = parameter("computeSharpness").toBool();
  _computeWarmth = parameter("computeWarmth").toBool();
  _computeBoominess = parameter("computeBoominess").toBool();
  _clipOutput = parameter("clipOutput").toBool();
  _loudnessNorm = parameter("loudnessNorm").toBool();
}

void AudioCommonsTimbralModels::reset() {
}

void AudioCommonsTimbralModels::compute() {
    const std::vector<Real>& signal = _signal.get();
    Real& brightness = _brightness.get();
    Real& depth = _depth.get();
    Real& hardness = _hardness.get();
    Real& roughness = _roughness.get();
    Real& sharpness = _sharpness.get();
    Real& warmth = _warmth.get(); 
    Real& boominess = _boominess.get();

    brightness = 0.0f;
    depth = 0.0f;
    hardness = 0.0f;
    roughness = 0.0f;
    sharpness = 0.0f;
    warmth = 0.0f;
    boominess = 0.0f;

    // Silence guard
    if (signal.empty()) {
        return;
    }

    Real maxAbsIn = 0.0f;
    for (Real s : signal) {
        Real a = std::abs(s);
        if (a > maxAbsIn) maxAbsIn = a;
    }
    if (maxAbsIn <= 0.0f) {
        return;
    }

    // Loudness normalization (-24 LUFS)
    std::vector<Real> audioNorm;
    if (_loudnessNorm) {
        audioNorm = TimbralDspHelpers::normalizeLoudnessEbuR128(signal, _sampleRate, -24.0f);
    } else {
        audioNorm = signal;
    }

    // Zwicker specific loudness framing
    bool needZwicker = _computeSharpness || _computeBoominess || _computeWarmth || _computeHardness;

    std::vector<Real> windowedRms;
    std::vector<Real> windowedSharpness;
    std::vector<Real> windowedBooming;
    std::vector<Real> boomBarkVals;
    std::vector<Real> wrVals;
    std::vector<Real> hfVals;
    std::vector<Real> midbandVals;

    if (needZwicker) {
        const size_t frameSize4096 = 4096;
        size_t rem = audioNorm.size() % frameSize4096;
        size_t paddedLen = (rem == 0) ? audioNorm.size() : (audioNorm.size() + (frameSize4096 - rem));
        if (paddedLen < frameSize4096) paddedLen = frameSize4096;

        size_t nFrames4096 = paddedLen / frameSize4096;

        SpecificLoudnessZwicker zwickerEngine;
        std::vector<Real> frameBuf(frameSize4096, 0.0f);
        std::vector<Real> specLoudness(240, 0.0f);

        for (size_t fi = 0; fi < nFrames4096; ++fi) {
            size_t start = fi * frameSize4096;
            Real sumSq = 0.0f;
            for (size_t k = 0; k < frameSize4096; ++k) {
                size_t idx = start + k;
                Real s = (idx < audioNorm.size()) ? audioNorm[idx] : 0.0f;
                frameBuf[k] = s;
                sumSq += s * s;
            }

            Real frameRms = std::sqrt(sumSq / static_cast<Real>(frameSize4096));
            windowedRms.push_back(frameRms);

            Real totalLoudness = 0.0f;
            zwickerEngine.compute(frameBuf, totalLoudness, specLoudness, 100.0f, 0);

            if (_computeSharpness) {
                Real sharpVal = (totalLoudness > 0.0f)
                                    ? SpecificLoudnessZwicker::computeSharpnessFastl(specLoudness)
                                    : 0.0f;
                windowedSharpness.push_back(sharpVal);
            }

            if (_computeBoominess) {
                Real frameBoom = 0.0f;
                Real lowBark = 0.0f;
                if (totalLoudness > 0.0f) {
                    SpecificLoudnessZwicker::computeBoominessMetrics(specLoudness, frameBoom, lowBark);
                }
                windowedBooming.push_back(frameBoom);
                boomBarkVals.push_back(lowBark);
            }

            if (_computeWarmth) {
                Real wrLevel = 0.0f;
                Real hfLevel = 0.0f;
                SpecificLoudnessZwicker::computeWarmthRegionLevels(specLoudness, wrLevel, hfLevel);
                wrVals.push_back(wrLevel);
                hfVals.push_back(hfLevel);
            }

            if (_computeHardness) {
                Real midLevel = SpecificLoudnessZwicker::computeHardnessMidbandLevel(specLoudness);
                midbandVals.push_back(midLevel);
            }
        }
    }

    // Sharpness
    if (_computeSharpness) {
        Real sumW = 0.0f;
        for (Real r : windowedRms) {
            sumW += r * r;
        }

        Real rmsSharpness = 0.0f;
        if (sumW > 0.0f) {
            for (size_t i = 0; i < windowedSharpness.size(); ++i) {
                rmsSharpness += windowedSharpness[i] * (windowedRms[i] * windowedRms[i]);
            }
            rmsSharpness /= sumW;
        }

        Real logSharp = std::log10(std::max(rmsSharpness, 1e-12f));
        Real sharpVal = logSharp * 102.50508921364404f + 34.432655185001735f;

        if (_clipOutput) {
            sharpVal = std::max(0.0f, std::min(100.0f, sharpVal));
        }
        sharpness = sharpVal;
    } else {
        sharpness = 0.0f;
    }

    // Boominess
    if (_computeBoominess) {
        Real meanBoomBark = (!boomBarkVals.empty())
                                ? (std::accumulate(boomBarkVals.begin(), boomBarkVals.end(), 0.0f) / boomBarkVals.size())
                                : 0.0f;
        Real ll = std::log10(std::max(meanBoomBark, 1e-12f));

        Real sumW = 0.0f;
        for (Real r : windowedRms) {
            sumW += r * r;
        }

        Real rmsBoom = 0.0f;
        if (sumW > 0.0f) {
            for (size_t i = 0; i < windowedBooming.size(); ++i) {
                rmsBoom += windowedBooming[i] * (windowedRms[i] * windowedRms[i]);
            }
            rmsBoom /= sumW;
        }
        Real logRmsBoom = std::log10(std::max(rmsBoom, 1e-12f));

        const Real boomCoeffs[3] = {
            43.67402696195865f,
            -10.90054738389845f,
            26.836530575185435f
        };

        Real boomVal = logRmsBoom * boomCoeffs[0] + ll * boomCoeffs[1] + boomCoeffs[2];

        if (_clipOutput) {
            boomVal = std::max(0.0f, std::min(100.0f, boomVal));
        }
        boominess = boomVal;
    } else {
        boominess = 0.0f;
    }

    // Brightness
    if (_computeBrightness) {
        std::vector<Real> sigMinFreq = TimbralDspHelpers::cascadeButterworthHighPass(signal, 20.0f, _sampleRate, 3, 1);
        std::vector<Real> sigRatioBr = TimbralDspHelpers::cascadeButterworthHighPass(sigMinFreq, 2000.0f, _sampleRate, 3, 1);
        std::vector<Real> sigCentroidBr = TimbralDspHelpers::cascadeButterworthHighPass(sigMinFreq, 100.0f, _sampleRate, 3, 1);

        Real maxAbs = 0.0f;
        for (Real s : sigMinFreq) {
            Real a = std::abs(s);
            if (a > maxAbs) maxAbs = a;
        }

        if (maxAbs <= 0.0f) {
            brightness = 0.0f;
        } else {
            for (size_t i = 0; i < sigMinFreq.size(); ++i) {
                sigMinFreq[i] /= maxAbs;
                sigRatioBr[i] /= maxAbs;
                sigCentroidBr[i] /= maxAbs;
            }

            const int winBr = 2048;
            const int hopBr = 512;

            if (sigMinFreq.size() < static_cast<size_t>(winBr)) {
                sigMinFreq.resize(winBr, 0.0f);
                sigRatioBr.resize(winBr, 0.0f);
                sigCentroidBr.resize(winBr, 0.0f);
            }

            int nFramesBr = static_cast<int>((sigMinFreq.size() - winBr) / hopBr) + 1;
            std::vector<Real> win = TimbralDspHelpers::hammingWindow(winBr);

            std::vector<Real> brRatios;
            std::vector<Real> brRatioWeights;
            std::vector<Real> brCentroids;
            std::vector<Real> brCentroidWeights;

            std::vector<Real> frameMin(winBr);
            std::vector<Real> frameRatio(winBr);
            std::vector<Real> frameCent(winBr);

            for (int fi = 0; fi < nFramesBr; ++fi) {
                int st = fi * hopBr;
                for (int k = 0; k < winBr; ++k) {
                    frameMin[k] = sigMinFreq[st + k] * win[k];
                    frameRatio[k] = sigRatioBr[st + k] * win[k];
                    frameCent[k] = sigCentroidBr[st + k] * win[k];
                }

                std::vector<Real> sMin = TimbralDspHelpers::powerSpectrum(frameMin, winBr);
                std::vector<Real> sRatio = TimbralDspHelpers::powerSpectrum(frameRatio, winBr);
                std::vector<Real> sCent = TimbralDspHelpers::powerSpectrum(frameCent, winBr);

                Real pAll = std::accumulate(sMin.begin(), sMin.end(), 0.0f);
                Real pRatio = std::accumulate(sRatio.begin(), sRatio.end(), 0.0f);
                Real pCent = std::accumulate(sCent.begin(), sCent.end(), 0.0f);

                if (pAll > 0.0f) {
                    brRatios.push_back(pRatio / pAll);
                    brRatioWeights.push_back(pAll);
                }

                if (pCent > 0.0f) {
                    Real cVal = TimbralDspHelpers::spectralCentroid(sCent, _sampleRate);
                    brCentroids.push_back(cVal);
                    brCentroidWeights.push_back(pCent);
                }
            }

            Real sumRatioW = std::accumulate(brRatioWeights.begin(), brRatioWeights.end(), 0.0f);
            Real R = 0.0f;
            if (sumRatioW > 0.0f) {
                for (size_t i = 0; i < brRatios.size(); ++i) {
                    R += brRatios[i] * brRatioWeights[i];
                }
                R /= sumRatioW;
            }

            Real sumCentW = std::accumulate(brCentroidWeights.begin(), brCentroidWeights.end(), 0.0f);
            Real C = 0.0f;
            if (sumCentW > 0.0f) {
                for (size_t i = 0; i < brCentroids.size(); ++i) {
                    C += brCentroids[i] * brCentroidWeights[i];
                }
                C /= sumCentW;
            }

            Real logR = (R > 0.0f) ? std::log10(R) : -12.0f;
            Real logC = (C > 0.0f) ? std::log10(C) : 0.0f;

            Real br = 4.6131280180f * logR + 17.3788893093f * logC + 17.4347337506f;
            if (_clipOutput) {
                br = std::max(0.0f, std::min(100.0f, br));
            }
            brightness = br;
        }
    } else {
        brightness = 0.0f;
    }

    // Depth
    if (_computeDepth) {
        std::vector<Real> sigHpDp = TimbralDspHelpers::cascadeButterworthHighPass(audioNorm, 20.0f, _sampleRate, 3, 1);
        std::vector<Real> sigCentDp = TimbralDspHelpers::cascadeButterworthLowPass(sigHpDp, 2000.0f, _sampleRate, 3, 1);
        std::vector<Real> sigRatioDp = TimbralDspHelpers::cascadeButterworthLowPass(sigHpDp, 500.0f, _sampleRate, 3, 1);

        Real peakHp = 0.0f;
        for (Real s : sigHpDp) {
            Real a = std::abs(s);
            if (a > peakHp) peakHp = a;
        }

        if (peakHp <= 0.0f) {
            depth = 0.0f;
        } else {
            for (size_t i = 0; i < sigHpDp.size(); ++i) {
                sigHpDp[i] /= peakHp;
                sigCentDp[i] /= peakHp;
                sigRatioDp[i] /= peakHp;
            }

            const int winDp = 4096;
            const int hopDp = 1024;
            std::vector<Real> win = TimbralDspHelpers::hammingWindow(winDp);

            if (sigHpDp.size() < static_cast<size_t>(winDp)) {
                sigHpDp.resize(winDp, 0.0f);
                sigCentDp.resize(winDp, 0.0f);
                sigRatioDp.resize(winDp, 0.0f);
            }

            int nFramesDp = static_cast<int>((sigHpDp.size() - winDp) / hopDp) + 1;

            std::vector<Real> pooled(winDp / 2 + 1, 0.0f);
            std::vector<Real> lowerCentroids;
            std::vector<Real> lowerRatios;
            std::vector<Real> weightsDp;

            std::vector<Real> fHp(winDp);
            std::vector<Real> fCent(winDp);
            std::vector<Real> fRatio(winDp);

            Real threshDp = std::pow(10.0f, -60.0f / 20.0f);

            for (int fi = 0; fi < nFramesDp; ++fi) {
                int st = fi * hopDp;
                for (int k = 0; k < winDp; ++k) {
                    fHp[k] = sigHpDp[st + k] * win[k];
                    fCent[k] = sigCentDp[st + k] * win[k];
                    fRatio[k] = sigRatioDp[st + k] * win[k];
                }

                std::vector<Real> sAll = TimbralDspHelpers::powerSpectrum(fHp, winDp);
                std::vector<Real> sCent = TimbralDspHelpers::powerSpectrum(fCent, winDp);
                std::vector<Real> sRatio = TimbralDspHelpers::powerSpectrum(fRatio, winDp);

                for (size_t k = 0; k < sAll.size(); ++k) {
                    pooled[k] += sAll[k];
                }

                Real tpower = std::accumulate(sAll.begin(), sAll.end(), 0.0f);
                weightsDp.push_back(tpower);

                if (tpower > threshDp) {
                    Real pCent = std::accumulate(sCent.begin(), sCent.end(), 0.0f);
                    Real cVal = (pCent > 0.0f) ? TimbralDspHelpers::spectralCentroid(sCent, _sampleRate) : 0.0f;
                    Real pRatio = std::accumulate(sRatio.begin(), sRatio.end(), 0.0f);
                    Real rVal = (tpower > 0.0f) ? (pRatio / tpower) : 0.0f;
                    lowerCentroids.push_back(cVal);
                    lowerRatios.push_back(rVal);
                } else {
                    lowerCentroids.push_back(0.0f);
                    lowerRatios.push_back(0.0f);
                }
            }

            Real wSumDp = std::accumulate(weightsDp.begin(), weightsDp.end(), 0.0f);
            Real metric1 = 0.0f;
            Real metric2 = 0.0f;
            if (wSumDp > 0.0f) {
                for (size_t i = 0; i < weightsDp.size(); ++i) {
                    metric1 += lowerCentroids[i] * weightsDp[i];
                    metric2 += lowerRatios[i] * weightsDp[i];
                }
                metric1 /= wSumDp;
                metric2 /= wSumDp;
            }
            metric1 = std::min(metric1, 2000.0f);

            // RMS decay duration
            const int rmsStep = 256;
            int nRmsFrames = (sigHpDp.size() >= static_cast<size_t>(rmsStep))
                                 ? static_cast<int>((sigHpDp.size() - rmsStep) / rmsStep) + 1
                                 : 0;
            std::vector<Real> rmsEnv;
            if (sigHpDp.size() < static_cast<size_t>(rmsStep)) {
                Real sumSq = 0.0f;
                for (Real s : sigHpDp) sumSq += s * s;
                rmsEnv.push_back(std::sqrt(sumSq / std::max(1UL, sigHpDp.size())));
            } else {
                for (int fi = 0; fi < nRmsFrames; ++fi) {
                    int st = fi * rmsStep;
                    Real sumSq = 0.0f;
                    for (int j = 0; j < rmsStep; ++j) {
                        Real s = sigHpDp[st + j];
                        sumSq += s * s;
                    }
                    rmsEnv.push_back(std::sqrt(sumSq / static_cast<Real>(rmsStep)));
                }
                size_t remIdx = nRmsFrames * rmsStep;
                if (remIdx < sigHpDp.size()) {
                    Real sumSq = 0.0f;
                    size_t remLen = sigHpDp.size() - remIdx;
                    for (size_t j = 0; j < remLen; ++j) {
                        Real s = sigHpDp[remIdx + j];
                        sumSq += s * s;
                    }
                    rmsEnv.push_back(std::sqrt(sumSq / static_cast<Real>(remLen)));
                }
            }

            Real maxRms = 0.0f;
            for (Real v : rmsEnv) {
                Real a = std::abs(v);
                if (a > maxRms) maxRms = a;
            }
            if (maxRms > 0.0f) {
                for (Real& v : rmsEnv) v /= maxRms;
            }

            std::vector<Real> envSh = TimbralOnsetEnvelope::sampleAndHoldEnvelope(sigHpDp, _sampleRate, 0.2f, 0.01f);
            std::vector<int> onsetsRaw;
            try {
                onsetsRaw = TimbralOnsetEnvelope::calculateOnsets(sigHpDp, envSh, _sampleRate, 20.0f, 300.0f, 10.0f, 10.0f, 100.0f, 4096);
                if (onsetsRaw.empty() || onsetsRaw.size() == 1) {
                    onsetsRaw = {0};
                }
            } catch (...) {
                onsetsRaw = {0};
            }

            Real timeConvert = _sampleRate / static_cast<Real>(rmsStep);
            Real decayThr = std::pow(10.0f, -4.0f / 20.0f);
            std::vector<Real> durations;

            for (size_t oi = 0; oi < onsetsRaw.size(); ++oi) {
                int onsetPt = onsetsRaw[oi] / rmsStep;
                int nextPt = (oi + 1 < onsetsRaw.size()) ? (onsetsRaw[oi + 1] / rmsStep) : static_cast<int>(rmsEnv.size());
                if (onsetPt >= static_cast<int>(rmsEnv.size())) continue;
                nextPt = std::min(nextPt, static_cast<int>(rmsEnv.size()));
                if (nextPt <= onsetPt) continue;

                int maxIdx = onsetPt;
                Real maxVal = rmsEnv[onsetPt];
                for (int k = onsetPt + 1; k < nextPt; ++k) {
                    if (rmsEnv[k] > maxVal) {
                        maxVal = rmsEnv[k];
                        maxIdx = k;
                    }
                }

                int postMaxLen = nextPt - maxIdx;
                if (postMaxLen <= 0) continue;

                Real minPost = rmsEnv[maxIdx];
                int underIdx = -1;
                for (int k = 0; k < postMaxLen; ++k) {
                    Real val = rmsEnv[maxIdx + k];
                    if (val < minPost) minPost = val;
                    if (underIdx == -1 && val < decayThr) {
                        underIdx = k;
                    }
                }

                Real dur = 0.0f;
                if (minPost >= decayThr) {
                    dur = static_cast<Real>(postMaxLen) / timeConvert;
                } else {
                    dur = static_cast<Real>(underIdx + 1) / timeConvert;
                }
                durations.push_back(dur);
            }

            Real meanDur = 0.0f;
            if (!durations.empty()) {
                meanDur = std::accumulate(durations.begin(), durations.end(), 0.0f) / durations.size();
            }
            Real metric3 = (meanDur > 0.0f) ? std::log10(std::max(meanDur, 1e-12f)) : 0.0f;

            // Lowest spectral peak
            Real pMin = pooled[0];
            Real pMax = pooled[0];
            for (Real v : pooled) {
                if (v < pMin) pMin = v;
                if (v > pMax) pMax = v;
            }

            Real metric4 = 0.0f;
            if (pMax > pMin) {
                std::vector<Real> normPooled(pooled.size());
                for (size_t k = 0; k < pooled.size(); ++k) {
                    normPooled[k] = (pooled[k] - pMin) / (pMax - pMin);
                }
                std::vector<Real> freqBins(pooled.size());
                Real binW = (_sampleRate * 0.5f) / static_cast<Real>(pooled.size() - 1);
                for (size_t k = 0; k < pooled.size(); ++k) {
                    freqBins[k] = k * binW;
                }
                auto [peakIdx, peakLevels, peakFreqs] = TimbralDspHelpers::detectPeaksMIR(normPooled, freqBins, 0.01f);
                (void)peakIdx;
                (void)peakLevels;
                if (!peakFreqs.empty() && peakFreqs[0] > 0.0f) {
                    Real minPf = peakFreqs[0];
                    for (Real pf : peakFreqs) {
                        if (pf < minPf) minPf = pf;
                    }
                    metric4 = std::log10(std::max(minPf, 1e-12f));
                }
            }

            Real metric5 = metric2 * metric3;
            Real metric6 = (1.0f / (1.0f + std::exp(-metric2))) * metric3;

            const Real depthCoeffs[7] = {
                -0.0043703565847874465f,
                32.83743202462131f,
                4.750862716905235f,
                -14.217438690256062f,
                3.8782339862813924f,
                -0.8544826091735516f,
                66.69534393444391f
            };

            Real depthVal = metric1 * depthCoeffs[0] +
                            metric2 * depthCoeffs[1] +
                            metric3 * depthCoeffs[2] +
                            metric4 * depthCoeffs[3] +
                            metric5 * depthCoeffs[4] +
                            metric6 * depthCoeffs[5] +
                            depthCoeffs[6];

            if (_clipOutput) {
                depthVal = std::max(0.0f, std::min(100.0f, depthVal));
            }
            depth = depthVal;
        }
    } else {
        depth = 0.0f;
    }

    // Roughness
    if (_computeRoughness) {
        roughness = TimbralDspHelpers::computeRoughness(audioNorm, _sampleRate, _clipOutput);
    } else {
        roughness = 0.0f;
    }

    // Warmth
    if (_computeWarmth) {
        Real meanWr = (!wrVals.empty())
                          ? (std::accumulate(wrVals.begin(), wrVals.end(), 0.0f) / wrVals.size())
                          : 0.0f;

        Real sumRms = std::accumulate(windowedRms.begin(), windowedRms.end(), 0.0f);
        Real weightedHf = 0.0f;
        if (sumRms > 0.0f) {
            for (size_t i = 0; i < hfVals.size(); ++i) {
                weightedHf += hfVals[i] * windowedRms[i];
            }
            weightedHf /= sumRms;
        }

        std::vector<Real> envWm = TimbralOnsetEnvelope::sampleAndHoldEnvelope(audioNorm, _sampleRate, 0.1f, 0.01f);
        std::vector<int> onsetsWm;
        try {
            onsetsWm = TimbralOnsetEnvelope::calculateOnsets(audioNorm, envWm, _sampleRate, 20.0f, 300.0f, 10.0f, 10.0f, 100.0f, 4096);
            if (onsetsWm.empty() || onsetsWm.size() == 1) {
                onsetsWm = {0};
            }
        } catch (...) {
            onsetsWm = {0};
        }

        const int maxFftWm = 8192;
        std::vector<Real> hannWin = TimbralDspHelpers::hannWindow(maxFftWm, false);
        std::vector<Real> allRmsWm;
        std::vector<Real> allRatioWm;
        std::vector<Real> allScWm;
        std::vector<Real> allDecayWm;

        Real binW = (_sampleRate * 0.5f) / static_cast<Real>(maxFftWm / 2);

        for (size_t idx = 0; idx < onsetsWm.size(); ++idx) {
            int onset = onsetsWm[idx];
            int nextOnset = (idx + 1 < onsetsWm.size()) ? onsetsWm[idx + 1] : static_cast<int>(audioNorm.size());
            if (onset >= static_cast<int>(audioNorm.size())) continue;
            nextOnset = std::min(nextOnset, static_cast<int>(audioNorm.size()));
            if (nextOnset <= onset) continue;

            int segLen = nextOnset - onset;
            Real sumSq = 0.0f;
            for (int k = 0; k < segLen; ++k) {
                Real s = audioNorm[onset + k];
                sumSq += s * s;
            }
            Real segRms = std::sqrt(sumSq / static_cast<Real>(segLen));
            allRmsWm.push_back(segRms);

            std::vector<Real> segSpec(maxFftWm / 2 + 1, 0.0f);
            if (segLen < maxFftWm) {
                std::vector<Real> segPad(maxFftWm, 0.0f);
                for (int k = 0; k < segLen; ++k) {
                    segPad[k] = audioNorm[onset + k] * hannWin[k];
                }
                segSpec = TimbralDspHelpers::powerSpectrum(segPad, maxFftWm);
            } else {
                int nSegFrames = segLen / maxFftWm;
                std::vector<Real> segFrame(maxFftWm);
                for (int sf = 0; sf < nSegFrames; ++sf) {
                    int sfStart = onset + sf * maxFftWm;
                    for (int k = 0; k < maxFftWm; ++k) {
                        segFrame[k] = audioNorm[sfStart + k] * hannWin[k];
                    }
                    std::vector<Real> pwr = TimbralDspHelpers::powerSpectrum(segFrame, maxFftWm);
                    for (size_t k = 0; k < segSpec.size(); ++k) {
                        segSpec[k] += pwr[k];
                    }
                }
                int remS = segLen % maxFftWm;
                if (remS > 0) {
                    std::fill(segFrame.begin(), segFrame.end(), 0.0f);
                    int sfStart = onset + nSegFrames * maxFftWm;
                    for (int k = 0; k < remS; ++k) {
                        segFrame[k] = audioNorm[sfStart + k] * hannWin[k];
                    }
                    std::vector<Real> pwr = TimbralDspHelpers::powerSpectrum(segFrame, maxFftWm);
                    for (size_t k = 0; k < segSpec.size(); ++k) {
                        segSpec[k] += pwr[k];
                    }
                }
            }

            Real maxSpec = 0.0f;
            for (Real v : segSpec) {
                if (v > maxSpec) maxSpec = v;
            }
            if (maxSpec > 0.0f) {
                for (Real& v : segSpec) v /= maxSpec;
            }

            // Fundamental frequency f0
            std::vector<int> peakIdx;
            std::vector<Real> peakFreq;
            for (size_t k = 1; k + 1 < segSpec.size(); ++k) {
                if (segSpec[k] > segSpec[k - 1] && segSpec[k] >= segSpec[k + 1] && segSpec[k] >= 0.01f) {
                    peakIdx.push_back(static_cast<int>(k));
                    peakFreq.push_back(static_cast<Real>(k) * binW);
                }
            }

            Real fundamental = 0.0f;
            int fundamentalIdx = 0;
            if (!peakFreq.empty()) {
                fundamental = peakFreq[0];
                fundamentalIdx = peakIdx[0];
                for (size_t p = 1; p < peakFreq.size(); ++p) {
                    if (peakFreq[p] < fundamental) {
                        fundamental = peakFreq[p];
                        fundamentalIdx = peakIdx[p];
                    }
                }
            } else {
                int argmax = 0;
                Real maxVal = segSpec[0];
                for (size_t k = 1; k < segSpec.size(); ++k) {
                    if (segSpec[k] > maxVal) {
                        maxVal = segSpec[k];
                        argmax = static_cast<int>(k);
                    }
                }
                fundamental = static_cast<Real>(argmax) * binW;
                fundamentalIdx = argmax;
            }

            // Warmth region energy ratio
            Real wrUpperF = std::min(fundamental * 3.5f, 12000.0f);
            Real tpower = std::accumulate(segSpec.begin(), segSpec.end(), 0.0f);
            int wrUpperIdx = static_cast<int>(segSpec.size()) - 1;
            for (size_t k = 0; k < segSpec.size(); ++k) {
                if (static_cast<Real>(k) * binW > wrUpperF) {
                    wrUpperIdx = static_cast<int>(k);
                    break;
                }
            }

            Real ratio = 0.0f;
            if (fundamental < 260.0f) {
                int topLevelIdx = static_cast<int>(segSpec.size()) - 1;
                for (size_t k = 0; k < segSpec.size(); ++k) {
                    if (static_cast<Real>(k) * binW > 260.0f) {
                        topLevelIdx = static_cast<int>(k);
                        break;
                    }
                }
                Real lowEnergy = 0.0f;
                for (int k = fundamentalIdx; k < topLevelIdx; ++k) {
                    lowEnergy += segSpec[k];
                }
                ratio = (tpower > 0.0f) ? (lowEnergy / tpower) : 0.0f;
            }
            allRatioWm.push_back(ratio);

            // Spectral centroid
            Real scNum = 0.0f;
            for (size_t k = 0; k < segSpec.size(); ++k) {
                scNum += (static_cast<Real>(k) * binW) * segSpec[k];
            }
            Real sc = (tpower > 0.0f) ? (scNum / tpower) : 0.0f;
            allScWm.push_back(sc);

            // HF decay score
            std::vector<Real> xLog;
            std::vector<Real> yLog;
            for (size_t k = wrUpperIdx; k < segSpec.size(); ++k) {
                Real f = static_cast<Real>(k) * binW;
                if (f > 0.0f && segSpec[k] > 0.0f) {
                    xLog.push_back(std::log10(f));
                    yLog.push_back(std::log10(segSpec[k]));
                }
            }
            Real decay = (xLog.size() >= 2) ? TimbralDspHelpers::r2LinearFit(xLog, yLog) : 0.0f;
            allDecayWm.push_back(decay);
        }

        Real meanSc = 0.0f;
        if (!allScWm.empty()) {
            Real avgSc = std::accumulate(allScWm.begin(), allScWm.end(), 0.0f) / allScWm.size();
            meanSc = std::log10(std::max(avgSc, 1e-12f));
        }

        Real meanDecayScore = (!allDecayWm.empty())
                                  ? (std::accumulate(allDecayWm.begin(), allDecayWm.end(), 0.0f) / allDecayWm.size())
                                  : 0.0f;

        Real sumRmsSeg = std::accumulate(allRmsWm.begin(), allRmsWm.end(), 0.0f);
        Real weightedMeanRatio = 0.0f;
        if (sumRmsSeg > 0.0f) {
            for (size_t i = 0; i < allRatioWm.size(); ++i) {
                weightedMeanRatio += allRatioWm[i] * allRmsWm[i];
            }
            weightedMeanRatio /= sumRmsSeg;
        }

        const Real warmthCoeffs[6] = {
            -4.464258317026696f,
            -0.08819320850778556f,
            0.29156539973575546f,
            17.274733561081554f,
            8.403340066029507f,
            45.21212125085579f
        };

        Real warmthVal = meanSc * warmthCoeffs[0] +
                         weightedHf * warmthCoeffs[1] +
                         meanWr * warmthCoeffs[2] +
                         meanDecayScore * warmthCoeffs[3] +
                         weightedMeanRatio * warmthCoeffs[4] +
                         warmthCoeffs[5];

        if (_clipOutput) {
            warmthVal = std::max(0.0f, std::min(100.0f, warmthVal));
        }
        warmth = warmthVal;
    } else {
        warmth = 0.0f;
    }

    // Hardness
    if (_computeHardness) {
        Real sumRms = std::accumulate(windowedRms.begin(), windowedRms.end(), 0.0f);
        Real weightedMidbandLevel = 0.0f;
        if (sumRms > 0.0f) {
            for (size_t i = 0; i < midbandVals.size(); ++i) {
                weightedMidbandLevel += midbandVals[i] * windowedRms[i];
            }
            weightedMidbandLevel /= sumRms;
        }
        Real logWeightedMidbandLevel = std::log10(std::max(weightedMidbandLevel, 1e-12f));

        Real hpRatio = TimbralDspHelpers::calculateHpssRatio(audioNorm, _sampleRate);
        Real logHpRatio = std::log10(std::max(hpRatio, 1e-12f));

        const int npersegHd = 4096;
        std::vector<Real> paddedAudioHd(audioNorm.size() + npersegHd + 1, 0.0f);
        std::copy(audioNorm.begin(), audioNorm.end(), paddedAudioHd.begin() + (npersegHd + 1));

        std::vector<Real> envelopeHd = TimbralOnsetEnvelope::sampleAndHoldEnvelope(paddedAudioHd, _sampleRate, 0.1f, 0.01f);
        std::vector<int> originalOnsetsHd;
        try {
            originalOnsetsHd = TimbralOnsetEnvelope::calculateOnsets(paddedAudioHd, envelopeHd, _sampleRate, 20.0f, 300.0f, 10.0f, 10.0f, 100.0f, npersegHd);
            if (originalOnsetsHd.empty() || originalOnsetsHd.size() == 1) {
                originalOnsetsHd = {0};
            }
        } catch (...) {
            originalOnsetsHd = {0};
        }

        std::vector<Real> onsetStrHd = TimbralOnsetEnvelope::onsetStrength(paddedAudioHd, _sampleRate, 2048, 512);

        std::vector<int> onsetsHd(originalOnsetsHd.size());
        for (size_t i = 0; i < originalOnsetsHd.size(); ++i) {
            int shifted = originalOnsetsHd[i] - npersegHd;
            onsetsHd[i] = std::max(0, shifted);
        }

        const int bwStep = 128;
        std::vector<Real> bandwidth = TimbralDspHelpers::calculateBandwidthArray(paddedAudioHd, _sampleRate, npersegHd, bwStep, -75.0f);
        Real bwFs = _sampleRate / static_cast<Real>(bwStep);

        std::vector<int> bwOnsets(onsetsHd.size());
        for (size_t i = 0; i < onsetsHd.size(); ++i) {
            bwOnsets[i] = onsetsHd[i] / bwStep;
        }

        std::vector<Real> allBwMax;
        std::vector<Real> allMaxStrBw;
        std::vector<Real> allAttTime;
        std::vector<Real> allMaxStr;
        std::vector<Real> allAttCentroid;

        for (size_t onsetCnt = 0; onsetCnt < bwOnsets.size(); ++onsetCnt) {
            int onset = bwOnsets[onsetCnt];
            int nextOnset = (onsetCnt + 1 < bwOnsets.size()) ? bwOnsets[onsetCnt + 1] : static_cast<int>(bandwidth.size());
            if (onset >= static_cast<int>(bandwidth.size())) continue;
            nextOnset = std::min(nextOnset, static_cast<int>(bandwidth.size()));
            if (nextOnset <= onset) continue;

            std::vector<Real> bwSeg(bandwidth.begin() + onset, bandwidth.begin() + nextOnset);
            Real maxBw = 0.0f;
            for (Real b : bwSeg) if (b > maxBw) maxBw = b;

            bool bwAttackFound = false;
            if (!bwSeg.empty() && maxBw > 0.0f) {
                auto [attTime, grad, stIdx] = TimbralDspHelpers::calculateAttackTime(bwSeg, bwFs, 0.1f, "fixed_threshold");
                (void)attTime;
                (void)grad;
                bwAttackFound = true;
                int maxAttSamples = static_cast<int>(0.1f * bwFs);
                int endIdx = std::min(static_cast<int>(bwSeg.size()), stIdx + maxAttSamples);
                Real localMax = bwSeg[stIdx];
                for (int k = stIdx; k < endIdx; ++k) {
                    if (bwSeg[k] > localMax) localMax = bwSeg[k];
                }
                allBwMax.push_back(localMax);
            }

            int origO = originalOnsetsHd[onsetCnt];
            int nextOrigO = (onsetCnt + 1 < originalOnsetsHd.size()) ? originalOnsetsHd[onsetCnt + 1] : static_cast<int>(envelopeHd.size());
            origO = std::min(origO, static_cast<int>(envelopeHd.size()));
            nextOrigO = std::min(nextOrigO, static_cast<int>(envelopeHd.size()));

            std::vector<Real> attSeg;
            if (origO < nextOrigO) {
                attSeg.assign(envelopeHd.begin() + origO, envelopeHd.begin() + nextOrigO);
            }

            int strStart = origO / 512;
            int strEnd = nextOrigO / 512;
            strStart = std::min(strStart, static_cast<int>(onsetStrHd.size()));
            strEnd = std::min(strEnd, static_cast<int>(onsetStrHd.size()));
            Real maxStr = 0.0f;
            for (int k = strStart; k < strEnd; ++k) {
                if (onsetStrHd[k] > maxStr) maxStr = onsetStrHd[k];
            }
            allMaxStr.push_back(maxStr);
            if (bwAttackFound) {
                allMaxStrBw.push_back(maxStr);
            }

            auto [attT, gradT, thSt] = TimbralDspHelpers::calculateAttackTime(attSeg, _sampleRate, 0.1f, "min_effort");
            (void)gradT;
            allAttTime.push_back(attT);

            int centSamples = static_cast<int>(0.125f * _sampleRate);
            int segLenHd = nextOrigO - origO;
            int audSegStart = origO + thSt;
            int audSegEnd = (thSt + centSamples >= segLenHd) ? nextOrigO : (audSegStart + centSamples);
            audSegEnd = std::min(audSegEnd, static_cast<int>(paddedAudioHd.size()));
            if (audSegStart < audSegEnd) {
                std::vector<Real> audSeg(paddedAudioHd.begin() + audSegStart, paddedAudioHd.begin() + audSegEnd);
                if (audSeg.size() > 512) {
                    Real cAtt = TimbralDspHelpers::calculateAttackCentroid(audSeg, _sampleRate);
                    if (cAtt > 0.0f) {
                        allAttCentroid.push_back(cAtt);
                    }
                }
            }
        }

        Real meanAttackTime = (!allAttTime.empty())
                                  ? (std::accumulate(allAttTime.begin(), allAttTime.end(), 0.0f) / allAttTime.size())
                                  : -2.0f;

        Real meanWeightedBwMax = _sampleRate / 512.0f;
        Real sumStrBw = std::accumulate(allMaxStrBw.begin(), allMaxStrBw.end(), 0.0f);
        if (!allBwMax.empty() && sumStrBw > 0.0f) {
            Real accBw = 0.0f;
            for (size_t i = 0; i < allBwMax.size(); ++i) {
                accBw += allBwMax[i] * allMaxStrBw[i];
            }
            meanWeightedBwMax = accBw / sumStrBw;
            if (meanWeightedBwMax <= 512.0f) {
                meanWeightedBwMax = _sampleRate / 512.0f;
            }
        }
        Real logWeightedBandwidthMax = std::log10(meanWeightedBwMax);

        Real meanMaxStrength = (!allMaxStr.empty())
                                   ? (std::accumulate(allMaxStr.begin(), allMaxStr.end(), 0.0f) / allMaxStr.size())
                                   : 1.0f;
        if (meanMaxStrength <= 0.0f) meanMaxStrength = 1.0f;
        Real logMeanMaxStrength = std::log10(meanMaxStrength);

        Real meanAttackCentroid = (!allAttCentroid.empty())
                                      ? (std::accumulate(allAttCentroid.begin(), allAttCentroid.end(), 0.0f) / allAttCentroid.size())
                                      : 200.0f;
        if (meanAttackCentroid <= 200.0f) meanAttackCentroid = 200.0f;
        Real logAttackCentroid = std::log10(meanAttackCentroid);

        const Real hardnessCoeffs[7] = {
            12.079781720638145f,
            18.52100377170042f,
            14.139883645260355f,
            5.567690321917516f,
            3.9346817690405635f,
            -4.326890461087848f,
            -85.60352209068202f
        };

        Real hardnessVal = logWeightedBandwidthMax * hardnessCoeffs[0] +
                           logAttackCentroid * hardnessCoeffs[1] +
                           logWeightedMidbandLevel * hardnessCoeffs[2] +
                           logHpRatio * hardnessCoeffs[3] +
                           logMeanMaxStrength * hardnessCoeffs[4] +
                           meanAttackTime * hardnessCoeffs[5] +
                           hardnessCoeffs[6];

        if (_clipOutput) {
            hardnessVal = std::max(0.0f, std::min(100.0f, hardnessVal));
        }
        hardness = hardnessVal;
    } else {
        hardness = 0.0f;
    }
}
