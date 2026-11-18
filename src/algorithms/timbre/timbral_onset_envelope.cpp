#include "timbral_onset_envelope.h"

#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace essentia {

namespace {

inline double hzToMel(double hz) {
    if (hz >= 1000.0) {
        double clamped = (hz < 1.0) ? 1.0 : hz;
        return 15.0 + std::log(clamped / 1000.0) * (27.0 / std::log(6.4));
    } else {
        return hz / (200.0 / 3.0);
    }
}

inline double melToHz(double mel) {
    if (mel >= 15.0) {
        return 1000.0 * std::exp((mel - 15.0) / (27.0 / std::log(6.4)));
    } else {
        return mel * (200.0 / 3.0);
    }
}

// 2048-point Radix-2 FFT
class FastFourierTransform2048 {
public:
    static constexpr int N = 2048;
    static constexpr int LOG2_N = 11;

    FastFourierTransform2048() {
        bitRev.resize(N);
        for (int i = 0; i < N; ++i) {
            int rev = 0;
            for (int b = 0; b < LOG2_N; ++b) {
                if ((i >> b) & 1) {
                    rev |= (1 << (LOG2_N - 1 - b));
                }
            }
            bitRev[static_cast<size_t>(i)] = rev;
        }

        cosTable.resize(N / 2);
        sinTable.resize(N / 2);
        for (int i = 0; i < N / 2; ++i) {
            double angle = -2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N);
            cosTable[static_cast<size_t>(i)] = std::cos(angle);
            sinTable[static_cast<size_t>(i)] = std::sin(angle);
        }
    }

    void powerSpectrum(const Real* input, Real* outPowerSpec) const {
        double re[N];
        double im[N];
        for (int i = 0; i < N; ++i) {
            int r = bitRev[static_cast<size_t>(i)];
            re[r] = static_cast<double>(input[i]);
            im[r] = 0.0;
        }

        // Cooley-Tukey decimation-in-time radix-2
        for (int len = 2; len <= N; len <<= 1) {
            int half = len >> 1;
            int step = N / len;
            for (int i = 0; i < N; i += len) {
                for (int j = 0; j < half; ++j) {
                    int k = j * step;
                    double c = cosTable[static_cast<size_t>(k)];
                    double s = sinTable[static_cast<size_t>(k)];
                    int uIdx = i + j;
                    int vIdx = i + j + half;
                    double vRe = re[vIdx] * c - im[vIdx] * s;
                    double vIm = re[vIdx] * s + im[vIdx] * c;
                    double uRe = re[uIdx];
                    double uIm = im[uIdx];
                    re[uIdx] = uRe + vRe;
                    im[uIdx] = uIm + vIm;
                    re[vIdx] = uRe - vRe;
                    im[vIdx] = uIm - vIm;
                }
            }
        }

        for (int k = 0; k <= N / 2; ++k) {
            outPowerSpec[k] = static_cast<Real>(re[k] * re[k] + im[k] * im[k]);
        }
    }

private:
    std::vector<int> bitRev;
    std::vector<double> cosTable;
    std::vector<double> sinTable;
};

} // anonymous namespace

std::vector<Real> TimbralOnsetEnvelope::sampleAndHoldEnvelope(
    const std::vector<Real>& audio,
    Real sampleRate,
    Real decayTime,
    Real holdTime)
{
    if (audio.empty()) {
        return {};
    }

    Real peak = 0.0f;
    for (Real s : audio) {
        Real a = std::fabs(s);
        if (a > peak) {
            peak = a;
        }
    }

    if (peak == 0.0f) {
        return std::vector<Real>(audio.size(), 0.0f);
    }

    Real decay = peak / (decayTime * sampleRate);
    int holdSamples = static_cast<int>(holdTime * sampleRate);
    std::vector<Real> env(audio.size(), 0.0f);
    Real prev = 0.0f;
    int holdCnt = 0;

    for (size_t j = 0; j < audio.size(); ++j) {
        Real s = std::fabs(audio[j]);
        if (s >= prev) {
            env[j] = s;
            prev = s;
            holdCnt = 0;
        } else if (holdCnt < holdSamples) {
            holdCnt++;
            env[j] = prev;
        } else {
            Real out = prev - decay;
            prev = std::max(out, s);
            env[j] = prev;
        }
    }

    return env;
}

TimbralOnsetEnvelope::MelFilterbank TimbralOnsetEnvelope::initMelFilterbank(
    Real sampleRate,
    int nFft,
    int nMels)
{
    MelFilterbank fb;
    fb.nMels = nMels;
    fb.nFft = nFft;

    int nBins = nFft / 2 + 1;
    fb.weights.assign(static_cast<size_t>(nMels * nBins), 0.0f);

    double melMin = hzToMel(0.0);
    double melMax = hzToMel(static_cast<double>(sampleRate) / 2.0);

    std::vector<double> freqs(static_cast<size_t>(nMels + 2));
    for (int i = 0; i < nMels + 2; ++i) {
        double m = melMin + static_cast<double>(i) * (melMax - melMin) / static_cast<double>(nMels + 1);
        freqs[static_cast<size_t>(i)] = melToHz(m);
    }

    std::vector<double> fftFreqs(static_cast<size_t>(nBins));
    for (int k = 0; k < nBins; ++k) {
        fftFreqs[static_cast<size_t>(k)] = static_cast<double>(k) * static_cast<double>(sampleRate) / static_cast<double>(nFft);
    }

    for (int i = 0; i < nMels; ++i) {
        double lower = freqs[static_cast<size_t>(i)];
        double center = freqs[static_cast<size_t>(i + 1)];
        double upper = freqs[static_cast<size_t>(i + 2)];
        double enorm = 2.0 / (upper - lower);

        for (int k = 0; k < nBins; ++k) {
            double f = fftFreqs[static_cast<size_t>(k)];
            double w = 0.0;
            if (f >= lower && f <= center && center > lower) {
                w = (f - lower) / (center - lower);
            }
            if (f >= center && f <= upper && upper > center) {
                w = (upper - f) / (upper - center);
            }
            fb.weights[static_cast<size_t>(i * nBins + k)] = static_cast<Real>(w * enorm);
        }
    }

    return fb;
}

const TimbralOnsetEnvelope::MelFilterbank& TimbralOnsetEnvelope::defaultMelFilterbank() {
    static const MelFilterbank instance = initMelFilterbank(44100.0f, 2048, 128);
    return instance;
}

std::vector<Real> TimbralOnsetEnvelope::onsetStrength(
    const std::vector<Real>& audio,
    Real sampleRate,
    int nFft,
    int hop)
{
    if (audio.empty()) {
        return {};
    }

    int pad = nFft / 2;
    std::vector<Real> yPad(audio.size() + static_cast<size_t>(2 * pad), 0.0f);
    std::copy(audio.begin(), audio.end(), yPad.begin() + pad);

    // Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2.0 * M_PI * n / nFft)
    std::vector<Real> w(static_cast<size_t>(nFft));
    for (int n = 0; n < nFft; ++n) {
        w[static_cast<size_t>(n)] = static_cast<Real>(
            0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(n) / static_cast<double>(nFft)));
    }

    // Number of frames matching np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop]
    size_t L_pad = yPad.size();
    if (L_pad < static_cast<size_t>(nFft)) {
        return {};
    }
    size_t nFrames = (L_pad - static_cast<size_t>(nFft)) / static_cast<size_t>(hop) + 1;
    if (nFrames == 0) {
        return {};
    }

    const MelFilterbank& mfb = (sampleRate == 44100.0f && nFft == 2048) ?
        defaultMelFilterbank() : initMelFilterbank(sampleRate, nFft, 128);
    int nMels = mfb.nMels;
    int nBins = nFft / 2 + 1;

    static const FastFourierTransform2048 fft2048;

    // Compute Mel spectrogram: mel[m][t]
    std::vector<Real> mel(static_cast<size_t>(nMels) * nFrames, 0.0f);
    std::vector<Real> frameW(static_cast<size_t>(nFft));
    std::vector<Real> pSpec(static_cast<size_t>(nBins));

    Real refVal = 0.0f;

    for (size_t t = 0; t < nFrames; ++t) {
        size_t start = t * static_cast<size_t>(hop);
        for (int n = 0; n < nFft; ++n) {
            frameW[static_cast<size_t>(n)] = yPad[start + static_cast<size_t>(n)] * w[static_cast<size_t>(n)];
        }

        fft2048.powerSpectrum(frameW.data(), pSpec.data());

        for (int m = 0; m < nMels; ++m) {
            const Real* wRow = &mfb.weights[static_cast<size_t>(m * nBins)];
            double sum = 0.0;
            for (int k = 0; k < nBins; ++k) {
                sum += static_cast<double>(wRow[k]) * static_cast<double>(pSpec[static_cast<size_t>(k)]);
            }
            Real melVal = static_cast<Real>(sum);
            mel[static_cast<size_t>(m) * nFrames + t] = melVal;
            if (melVal > refVal) {
                refVal = melVal;
            }
        }
    }

    // Convert mel to dB: 10 * log10(max(mel, 1e-10)) - 10 * log10(ref_val), floored at -80 dB
    double logRef = (refVal > 0.0f) ? (10.0 * std::log10(static_cast<double>(refVal))) : 0.0;
    std::vector<Real> melDb(static_cast<size_t>(nMels) * nFrames);

    for (size_t idx = 0; idx < mel.size(); ++idx) {
        double val = static_cast<double>(mel[idx]);
        if (val < 1e-10) {
            val = 1e-10;
        }
        double db = 10.0 * std::log10(val) - logRef;
        if (db < -80.0) {
            db = -80.0;
        }
        melDb[idx] = static_cast<Real>(db);
    }

    if (nFrames <= 1) {
        return std::vector<Real>(nFrames, 0.0f);
    }

    // Temporal flux: diff[m, t] = max(0.0, mel_db[m, t+1] - mel_db[m, t])
    // mean_flux[t] = mean over m
    std::vector<Real> meanFlux(nFrames - 1, 0.0f);
    for (size_t t = 0; t < nFrames - 1; ++t) {
        double fluxSum = 0.0;
        for (int m = 0; m < nMels; ++m) {
            Real curr = melDb[static_cast<size_t>(m) * nFrames + (t + 1)];
            Real prev = melDb[static_cast<size_t>(m) * nFrames + t];
            Real d = curr - prev;
            if (d > 0.0f) {
                fluxSum += static_cast<double>(d);
            }
        }
        meanFlux[t] = static_cast<Real>(fluxSum / static_cast<double>(nMels));
    }

    // Pad front with padWidth = 1 + nFft / (2 * hop) = 1 + 2048 / 1024 = 3 zeros
    // Result sliced to nFrames
    int padWidth = 1 + nFft / (2 * hop);
    std::vector<Real> result(nFrames, 0.0f);
    for (size_t t = 0; t < meanFlux.size(); ++t) {
        size_t targetIdx = t + static_cast<size_t>(padWidth);
        if (targetIdx < nFrames) {
            result[targetIdx] = meanFlux[t];
        }
    }

    return result;
}

std::vector<int> TimbralOnsetEnvelope::peakPick(
    const std::vector<Real>& x,
    int preMax,
    int postMax,
    int preAvg,
    int postAvg,
    Real delta,
    int wait)
{
    std::vector<int> peaks;
    int len = static_cast<int>(x.size());
    if (len == 0) {
        return peaks;
    }

    // Evaluate peak at index 0
    bool peak0 = true;
    int endPostMax0 = std::min(postMax, len);
    for (int i = 0; i < endPostMax0; ++i) {
        if (x[0] < x[static_cast<size_t>(i)]) {
            peak0 = false;
            break;
        }
    }
    if (peak0) {
        int endPostAvg0 = std::min(postAvg, len);
        double sum0 = 0.0;
        for (int i = 0; i < endPostAvg0; ++i) {
            sum0 += static_cast<double>(x[static_cast<size_t>(i)]);
        }
        double avg0 = (endPostAvg0 > 0) ? (sum0 / static_cast<double>(endPostAvg0)) : 0.0;
        if (x[0] < static_cast<Real>(avg0 + static_cast<double>(delta))) {
            peak0 = false;
        }
    }

    if (peak0) {
        peaks.push_back(0);
    }

    int n = peak0 ? (wait + 1) : 1;
    while (n < len) {
        int startMax = std::max(0, n - preMax);
        int endMax = std::min(n + postMax, len);
        Real maxn = x[static_cast<size_t>(startMax)];
        for (int i = startMax + 1; i < endMax; ++i) {
            if (x[static_cast<size_t>(i)] > maxn) {
                maxn = x[static_cast<size_t>(i)];
            }
        }

        if (x[static_cast<size_t>(n)] != maxn) {
            n += 1;
            continue;
        }

        int startAvg = std::max(0, n - preAvg);
        int endAvg = std::min(n + postAvg, len);
        double sumAvg = 0.0;
        for (int i = startAvg; i < endAvg; ++i) {
            sumAvg += static_cast<double>(x[static_cast<size_t>(i)]);
        }
        double avgn = (endAvg > startAvg) ? (sumAvg / static_cast<double>(endAvg - startAvg)) : 0.0;

        if (static_cast<double>(x[static_cast<size_t>(n)]) >= avgn + static_cast<double>(delta)) {
            peaks.push_back(n);
            n += wait + 1;
        } else {
            n += 1;
        }
    }

    return peaks;
}

std::vector<int> TimbralOnsetEnvelope::onsetBacktrack(
    const std::vector<int>& events,
    const std::vector<Real>& energy)
{
    if (events.empty()) {
        return {};
    }

    // Find local minima: (energy[t] <= energy[t - 1]) && (energy[t] < energy[t + 1])
    std::vector<int> minima;
    minima.push_back(0);
    int n = static_cast<int>(energy.size());
    for (int t = 1; t < n - 1; ++t) {
        if (energy[static_cast<size_t>(t)] <= energy[static_cast<size_t>(t - 1)] &&
            energy[static_cast<size_t>(t)] < energy[static_cast<size_t>(t + 1)]) {
            minima.push_back(t);
        }
    }

    std::vector<int> results;
    results.reserve(events.size());
    for (int ev : events) {
        auto it = std::upper_bound(minima.begin(), minima.end(), ev);
        if (it != minima.begin()) {
            results.push_back(*(it - 1));
        } else {
            results.push_back(0);
        }
    }

    return results;
}

std::vector<int> TimbralOnsetEnvelope::onsetDetect(
    const std::vector<Real>& audio,
    Real sampleRate,
    int hop)
{
    std::vector<Real> oenv = onsetStrength(audio, sampleRate, 2048, hop);
    if (oenv.empty()) {
        return {};
    }

    Real minVal = oenv[0];
    Real maxVal = oenv[0];
    for (Real v : oenv) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    Real dyn = maxVal - minVal;
    if (dyn <= 0.0f) {
        return {};
    }

    std::vector<Real> oenvNorm(oenv.size());
    for (size_t i = 0; i < oenv.size(); ++i) {
        oenvNorm[i] = (oenv[i] - minVal) / dyn;
    }

    int preMax = static_cast<int>(0.03f * sampleRate / static_cast<float>(hop));
    int postMax = static_cast<int>(0.00f * sampleRate / static_cast<float>(hop) + 1.0f);
    int preAvg = static_cast<int>(0.10f * sampleRate / static_cast<float>(hop));
    int postAvg = static_cast<int>(0.10f * sampleRate / static_cast<float>(hop) + 1.0f);
    int wait = static_cast<int>(0.03f * sampleRate / static_cast<float>(hop));
    Real delta = 0.07f;

    std::vector<int> peaks = peakPick(oenvNorm, preMax, postMax, preAvg, postAvg, delta, wait);
    std::vector<int> backtracked = onsetBacktrack(peaks, oenvNorm);
    for (int& b : backtracked) {
        b *= hop;
    }

    return backtracked;
}

int TimbralOnsetEnvelope::returnLoop(
    int onsetLoc,
    const std::vector<Real>& envelope,
    int functionTimeThresh,
    Real histThreshold,
    int histTimeSamples,
    int nperseg)
{
    int envSize = static_cast<int>(envelope.size());
    if (onsetLoc < 0 || onsetLoc >= envSize) {
        return 0;
    }

    while (true) {
        if (onsetLoc < 0 || onsetLoc >= envSize) {
            return 0;
        }
        Real currentSample = envelope[static_cast<size_t>(onsetLoc)];

        int evalStart = 0;
        int evalEnd = 0;
        if (onsetLoc - functionTimeThresh > 0) {
            evalStart = onsetLoc - functionTimeThresh - 1;
            evalEnd = onsetLoc;
        } else {
            evalStart = 0;
            evalEnd = onsetLoc - 1;
        }

        if (evalStart < 0) evalStart = 0;
        if (evalEnd > envSize) evalEnd = envSize;
        if (evalEnd <= evalStart) {
            return 0;
        }

        Real minEval = envelope[static_cast<size_t>(evalStart)];
        int minIdx = 0;
        for (int k = evalStart; k < evalEnd; ++k) {
            if (envelope[static_cast<size_t>(k)] < minEval) {
                minEval = envelope[static_cast<size_t>(k)];
                minIdx = k - evalStart;
            }
        }

        if (minEval - currentSample <= 0.0f) {
            int newOnsetLoc = minIdx + onsetLoc - functionTimeThresh - 1;
            if (newOnsetLoc > nperseg) {
                onsetLoc = newOnsetLoc;
            } else {
                return 0;
            }
        } else {
            int hystStart = 0;
            int hystEnd = onsetLoc;
            if (onsetLoc - histTimeSamples - 1 > 0) {
                hystStart = onsetLoc - histTimeSamples - 1;
            } else {
                hystStart = 0;
            }
            if (hystStart < 0) hystStart = 0;
            if (hystEnd > envSize) hystEnd = envSize;

            int lastMin = -1;
            for (int k = hystStart; k < hystEnd; ++k) {
                if (envelope[static_cast<size_t>(k)] < envelope[static_cast<size_t>(onsetLoc)]) {
                    lastMin = k - hystStart;
                }
            }

            if (lastMin == -1) {
                return onsetLoc;
            }

            int hystLen = hystEnd - hystStart;
            int lastIdx = onsetLoc - hystLen + lastMin;

            int segStart = hystStart + lastMin;
            Real segMax = envelope[static_cast<size_t>(segStart)];
            Real segMin = envelope[static_cast<size_t>(segStart)];
            for (int k = segStart; k < hystEnd; ++k) {
                Real val = envelope[static_cast<size_t>(k)];
                if (val > segMax) segMax = val;
                if (val < segMin) segMin = val;
            }

            Real segmentDynamicRange = segMax - segMin;
            if (segmentDynamicRange >= histThreshold) {
                return onsetLoc;
            } else {
                if (lastIdx >= nperseg) {
                    onsetLoc = lastIdx;
                } else {
                    return 0;
                }
            }
        }
    }
}

std::vector<int> TimbralOnsetEnvelope::calculateOnsets(
    const std::vector<Real>& audio,
    const std::vector<Real>& envelope,
    Real sampleRate,
    Real lookBackTime,
    Real hysteresisTime,
    Real hysteresisPercent,
    Real onsetInNoiseThreshold,
    Real minOnsetSeparationMs,
    int nperseg)
{
    std::vector<int> onsets = onsetDetect(audio, sampleRate, 512);
    if (onsets.empty()) {
        return {0};
    }

    int timeThresh = static_cast<int>(lookBackTime * 0.001f * sampleRate);
    int hysteresisSamples = static_cast<int>(hysteresisTime * 0.001f * sampleRate);

    if (envelope.empty()) {
        return {0};
    }

    Real envMin = envelope[0];
    Real envMax = envelope[0];
    for (Real v : envelope) {
        if (v < envMin) envMin = v;
        if (v > envMax) envMax = v;
    }
    Real envelopeDynRange = envMax - envMin;
    Real hysteresisThresh = envelopeDynRange * hysteresisPercent * 0.01f;

    std::vector<int> correctedOnsets;
    for (int onsetIdx : onsets) {
        if (onsetIdx > 0) {
            if (correctedOnsets.empty()) {
                int onsetHold = returnLoop(
                    onsetIdx, envelope, timeThresh, hysteresisThresh, hysteresisSamples, nperseg);
                correctedOnsets.push_back(onsetHold);
            } else {
                if (onsetIdx + 511 < static_cast<int>(envelope.size())) {
                    int onsetHold = returnLoop(
                        onsetIdx, envelope, timeThresh, hysteresisThresh, hysteresisSamples, nperseg);
                    correctedOnsets.push_back(onsetHold);
                }
            }
        } else {
            correctedOnsets.push_back(0);
        }
    }

    // Zero handling: keep index 0 if it is 0; remove any downstream 0s
    std::vector<int> nonZeroOnsets;
    for (size_t i = 0; i < correctedOnsets.size(); ++i) {
        if (correctedOnsets[i] == 0) {
            if (i == 0) {
                nonZeroOnsets.push_back(0);
            }
        } else {
            nonZeroOnsets.push_back(correctedOnsets[i]);
        }
    }
    correctedOnsets = std::move(nonZeroOnsets);

    // Deduplication preserving original order
    std::vector<int> uniqueOnsets;
    for (int val : correctedOnsets) {
        if (std::find(uniqueOnsets.begin(), uniqueOnsets.end(), val) == uniqueOnsets.end()) {
            uniqueOnsets.push_back(val);
        }
    }
    correctedOnsets = std::move(uniqueOnsets);

    // Onset in noise filtering: check dynamic range from onset to peak
    if (correctedOnsets.size() > 1) {
        Real threshold = onsetInNoiseThreshold * envelopeDynRange * 0.01f;
        for (int i = static_cast<int>(correctedOnsets.size()) - 1; i >= 0; --i) {
            int startIdx = correctedOnsets[static_cast<size_t>(i)];
            int endIdx = (i == static_cast<int>(correctedOnsets.size()) - 1) ?
                static_cast<int>(envelope.size()) : correctedOnsets[static_cast<size_t>(i + 1)];

            if (startIdx < 0) startIdx = 0;
            if (endIdx > static_cast<int>(envelope.size())) endIdx = static_cast<int>(envelope.size());

            int segLen = endIdx - startIdx;
            if (segLen > 1) {
                int peakRelIdx = 0;
                Real segMax = envelope[static_cast<size_t>(startIdx)];
                for (int k = 1; k < segLen; ++k) {
                    Real val = envelope[static_cast<size_t>(startIdx + k)];
                    if (val > segMax) {
                        segMax = val;
                        peakRelIdx = k;
                    }
                }

                if (peakRelIdx > 0) {
                    Real minBeforePeak = envelope[static_cast<size_t>(startIdx)];
                    for (int k = 1; k < peakRelIdx; ++k) {
                        Real val = envelope[static_cast<size_t>(startIdx + k)];
                        if (val < minBeforePeak) {
                            minBeforePeak = val;
                        }
                    }
                    Real segDynRange = segMax - minBeforePeak;
                    if (segDynRange < threshold) {
                        correctedOnsets.erase(correctedOnsets.begin() + i);
                    }
                } else {
                    correctedOnsets.erase(correctedOnsets.begin() + i);
                }
            } else {
                correctedOnsets.erase(correctedOnsets.begin() + i);
            }
        }
    }

    // Minimum temporal separation filtering
    if (correctedOnsets.size() > 1) {
        Real minSepSamples = sampleRate * 0.001f * minOnsetSeparationMs;
        while (correctedOnsets.size() > 1) {
            std::vector<int> toRemove;
            for (size_t i = 0; i < correctedOnsets.size() - 1; ++i) {
                if (std::abs(correctedOnsets[i + 1] - correctedOnsets[i]) < minSepSamples) {
                    toRemove.push_back(static_cast<int>(i + 1));
                }
            }
            if (toRemove.empty()) {
                break;
            }
            std::sort(toRemove.begin(), toRemove.end());
            toRemove.erase(std::unique(toRemove.begin(), toRemove.end()), toRemove.end());
            for (int k = static_cast<int>(toRemove.size()) - 1; k >= 0; --k) {
                correctedOnsets.erase(correctedOnsets.begin() + toRemove[static_cast<size_t>(k)]);
            }
        }
    }

    // Peak onset strength thresholding (>= 3.0)
    std::vector<Real> onsetStr = onsetStrength(audio, sampleRate, 2048, 512);
    std::vector<int> strengthOnsetTimes(correctedOnsets.size());
    for (size_t i = 0; i < correctedOnsets.size(); ++i) {
        int t = correctedOnsets[i] / 512;
        strengthOnsetTimes[i] = (t < 0) ? 0 : t;
    }

    std::vector<int> thdCorrectedOnsets;
    int nCorr = static_cast<int>(correctedOnsets.size());
    for (int onsetIdx = nCorr - 1; onsetIdx >= 0; --onsetIdx) {
        int currentStrength = strengthOnsetTimes[static_cast<size_t>(onsetIdx)];
        int startStr = currentStrength;
        int endStr = (currentStrength == strengthOnsetTimes.back()) ?
            static_cast<int>(onsetStr.size()) : strengthOnsetTimes[static_cast<size_t>(onsetIdx + 1)];

        if (startStr < 0) startStr = 0;
        if (endStr > static_cast<int>(onsetStr.size())) endStr = static_cast<int>(onsetStr.size());

        Real maxStr = -1e9f;
        bool hasStr = false;
        for (int k = startStr; k < endStr; ++k) {
            Real val = onsetStr[static_cast<size_t>(k)];
            if (val > maxStr) maxStr = val;
            hasStr = true;
        }

        if (hasStr && maxStr >= 3.0f) {
            thdCorrectedOnsets.push_back(correctedOnsets[static_cast<size_t>(onsetIdx)]);
        }
    }

    std::sort(thdCorrectedOnsets.begin(), thdCorrectedOnsets.end());
    if (!thdCorrectedOnsets.empty()) {
        return thdCorrectedOnsets;
    }
    return {0};
}

} // namespace essentia
