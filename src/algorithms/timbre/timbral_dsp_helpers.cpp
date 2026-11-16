#include "timbral_dsp_helpers.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace essentia {

Real TimbralDspHelpers::measureIntegratedLoudnessEbuR128(const std::vector<Real>& audio,
                                                        Real sampleRate) {
    if (audio.empty()) {
        return -70.0f;
    }

    const double fs = static_cast<double>(sampleRate);

    // Stage 1: High shelf filter (DeMan compliant weighting)
    // f0 = 1681.9744509555319 Hz, G = 3.99984385397 dB, Q = 0.7071752369554193
    const double K1 = std::tan(M_PI * 1681.9744509555319 / fs);
    const double Vh = std::pow(10.0, 3.99984385397 / 20.0);
    const double Vb = std::pow(Vh, 0.499666774155);
    const double Q1 = 0.7071752369554193;

    const double a0_1 = 1.0 + K1 / Q1 + K1 * K1;
    const double b0_1 = (Vh + Vb * K1 / Q1 + K1 * K1) / a0_1;
    const double b1_1 = 2.0 * (K1 * K1 - Vh) / a0_1;
    const double b2_1 = (Vh - Vb * K1 / Q1 + K1 * K1) / a0_1;
    const double a1_1 = 2.0 * (K1 * K1 - 1.0) / a0_1;
    const double a2_1 = (1.0 - K1 / Q1 + K1 * K1) / a0_1;

    // Stage 2: High-pass RLB weighting filter
    // f0 = 38.13547087613982 Hz, Q = 0.5003270373253953
    const double K2 = std::tan(M_PI * 38.13547087613982 / fs);
    const double Q2 = 0.5003270373253953;

    const double a0_2 = 1.0 + K2 / Q2 + K2 * K2;
    const double b0_2 = 1.0 / a0_2;
    const double b1_2 = -2.0 / a0_2;
    const double b2_2 = 1.0 / a0_2;
    const double a1_2 = 2.0 * (K2 * K2 - 1.0) / a0_2;
    const double a2_2 = (1.0 - K2 / Q2 + K2 * K2) / a0_2;

    const size_t N = audio.size();
    std::vector<double> y1(N);
    double d1 = 0.0;
    double d2 = 0.0;
    for (size_t n = 0; n < N; ++n) {
        const double xn = static_cast<double>(audio[n]);
        const double yn = b0_1 * xn + d1;
        d1 = b1_1 * xn - a1_1 * yn + d2;
        d2 = b2_1 * xn - a2_1 * yn;
        y1[n] = yn;
    }

    std::vector<double> y2(N);
    d1 = 0.0;
    d2 = 0.0;
    for (size_t n = 0; n < N; ++n) {
        const double xn = y1[n];
        const double yn = b0_2 * xn + d1;
        d1 = b1_2 * xn - a1_2 * yn + d2;
        d2 = b2_2 * xn - a2_2 * yn;
        y2[n] = yn;
    }

    // Framing: 400 ms blocks with 75% overlap (100 ms hop step)
    const size_t L = static_cast<size_t>(std::round(0.400 * fs));
    const size_t H = static_cast<size_t>(std::round(0.100 * fs));

    if (L == 0 || H == 0) {
        return -70.0f;
    }

    std::vector<double> z;
    if (N < L) {
        double sumSq = 0.0;
        for (size_t n = 0; n < N; ++n) {
            sumSq += y2[n] * y2[n];
        }
        z.push_back(sumSq / static_cast<double>(N));
    } else {
        const size_t numBlocks = (N > L) ? ((N - L - 1) / H + 2) : 1;
        z.reserve(numBlocks);
        for (size_t j = 0; j < numBlocks; ++j) {
            const size_t start = j * H;
            double sumSq = 0.0;
            for (size_t k = 0; k < L; ++k) {
                const size_t idx = start + k;
                const double s = (idx < N) ? y2[idx] : 0.0;
                sumSq += s * s;
            }
            z.push_back(sumSq / static_cast<double>(L));
        }
    }

    if (z.empty()) {
        return -70.0f;
    }

    // Absolute gating at -70 LKFS (Gamma_a)
    // l_j = -0.691 + 10 * log10(z_j) >= -70.0  =>  z_j >= 10^((-70 + 0.691)/10)
    const double zAbsThresh = std::pow(10.0, (-70.0 + 0.691) / 10.0);
    std::vector<double> zAbs;
    zAbs.reserve(z.size());
    for (double zj : z) {
        if (zj >= zAbsThresh) {
            zAbs.push_back(zj);
        }
    }

    if (zAbs.empty()) {
        return -70.0f;
    }

    double sumAbs = 0.0;
    for (double zj : zAbs) {
        sumAbs += zj;
    }
    const double zAbsMean = sumAbs / static_cast<double>(zAbs.size());
    // Relative threshold: Gamma_r = -0.691 + 10 * log10(zAbsMean) - 10.0 LU
    // Equivalent power threshold: zRelThresh = zAbsMean * 0.1
    const double zRelThresh = 0.1 * zAbsMean;

    std::vector<double> zRel;
    zRel.reserve(z.size());
    for (double zj : z) {
        if (zj > zRelThresh && zj >= zAbsThresh) {
            zRel.push_back(zj);
        }
    }

    if (zRel.empty()) {
        return -70.0f;
    }

    double sumRel = 0.0;
    for (double zj : zRel) {
        sumRel += zj;
    }
    const double zRelMean = sumRel / static_cast<double>(zRel.size());
    const double Lint = -0.691 + 10.0 * std::log10(std::max(zRelMean, 1e-12));

    return static_cast<Real>(Lint);
}

std::vector<Real> TimbralDspHelpers::normalizeLoudnessEbuR128(const std::vector<Real>& audio,
                                                             Real sampleRate,
                                                             Real targetLufs) {
    if (audio.empty()) {
        return audio;
    }

    const Real currentLufs = measureIntegratedLoudnessEbuR128(audio, sampleRate);
    if (currentLufs <= -69.9f) {
        // Essentially silent; leave unchanged
        return audio;
    }

    const double gain = std::pow(10.0, static_cast<double>(targetLufs - currentLufs) / 20.0);
    const Real g = static_cast<Real>(gain);

    std::vector<Real> out(audio.size());
    Real maxAbs = 0.0f;
    for (size_t i = 0; i < audio.size(); ++i) {
        out[i] = audio[i] * g;
        const Real absVal = std::abs(out[i]);
        if (absVal > maxAbs) {
            maxAbs = absVal;
        }
    }

    // Peak limit to avoid clipping if normalized output exceeds 1.0
    if (maxAbs > 1.0f) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] /= maxAbs;
        }
    }

    return out;
}

std::vector<Real> TimbralDspHelpers::cascadeButterworthHighPass(const std::vector<Real>& signal,
                                                               Real cutoffFreq,
                                                               Real sampleRate,
                                                               int orderStages,
                                                               int biquadOrder) {
    if (signal.empty() || orderStages <= 0) {
        return signal;
    }

    const double fs = static_cast<double>(sampleRate);
    const double fc = static_cast<double>(cutoffFreq);

    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

    if (biquadOrder == 1) {
        // Essentia 1st-order IIR HighPass (bilinear transform, Zölzer DAFX p. 40)
        const double K = std::tan(M_PI * fc / fs);
        b0 = 1.0 / (1.0 + K);
        b1 = -b0;
        b2 = 0.0;
        a1 = -(1.0 - K) / (1.0 + K);
        a2 = 0.0;
    } else {
        // 2nd-order Butterworth biquad
        const double w0 = 2.0 * M_PI * fc / fs;
        const double alpha = std::sin(w0) / std::sqrt(2.0);
        const double a0 = 1.0 + alpha;
        b0 = ((1.0 + std::cos(w0)) / 2.0) / a0;
        b1 = (-(1.0 + std::cos(w0))) / a0;
        b2 = ((1.0 + std::cos(w0)) / 2.0) / a0;
        a1 = (-2.0 * std::cos(w0)) / a0;
        a2 = (1.0 - alpha) / a0;
    }

    std::vector<Real> current = signal;
    for (int s = 0; s < orderStages; ++s) {
        std::vector<Real> next(current.size());
        double d1 = 0.0;
        double d2 = 0.0;
        for (size_t n = 0; n < current.size(); ++n) {
            const double xn = static_cast<double>(current[n]);
            const double yn = b0 * xn + d1;
            d1 = b1 * xn - a1 * yn + d2;
            d2 = b2 * xn - a2 * yn;
            next[n] = static_cast<Real>(yn);
        }
        current = std::move(next);
    }

    return current;
}

std::vector<Real> TimbralDspHelpers::cascadeButterworthLowPass(const std::vector<Real>& signal,
                                                              Real cutoffFreq,
                                                              Real sampleRate,
                                                              int orderStages,
                                                              int biquadOrder) {
    if (signal.empty() || orderStages <= 0) {
        return signal;
    }

    const double fs = static_cast<double>(sampleRate);
    const double fc = static_cast<double>(cutoffFreq);

    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

    if (biquadOrder == 1) {
        // Essentia 1st-order IIR LowPass (bilinear transform, Zölzer DAFX p. 40)
        const double K = std::tan(M_PI * fc / fs);
        b0 = K / (1.0 + K);
        b1 = b0;
        b2 = 0.0;
        a1 = -(1.0 - K) / (1.0 + K);
        a2 = 0.0;
    } else {
        // 2nd-order Butterworth biquad
        const double w0 = 2.0 * M_PI * fc / fs;
        const double alpha = std::sin(w0) / std::sqrt(2.0);
        const double a0 = 1.0 + alpha;
        b0 = ((1.0 - std::cos(w0)) / 2.0) / a0;
        b1 = (1.0 - std::cos(w0)) / a0;
        b2 = ((1.0 - std::cos(w0)) / 2.0) / a0;
        a1 = (-2.0 * std::cos(w0)) / a0;
        a2 = (1.0 - alpha) / a0;
    }

    std::vector<Real> current = signal;
    for (int s = 0; s < orderStages; ++s) {
        std::vector<Real> next(current.size());
        double d1 = 0.0;
        double d2 = 0.0;
        for (size_t n = 0; n < current.size(); ++n) {
            const double xn = static_cast<double>(current[n]);
            const double yn = b0 * xn + d1;
            d1 = b1 * xn - a1 * yn + d2;
            d2 = b2 * xn - a2 * yn;
            next[n] = static_cast<Real>(yn);
        }
        current = std::move(next);
    }

    return current;
}

std::vector<Real> TimbralDspHelpers::hammingWindow(int size) {
    std::vector<Real> w(size);
    if (size <= 1) {
        std::fill(w.begin(), w.end(), 1.0f);
        return w;
    }
    for (int n = 0; n < size; ++n) {
        w[n] = static_cast<Real>(0.54 - 0.46 * std::cos(2.0 * M_PI * n / (size - 1)));
    }
    return w;
}

std::vector<Real> TimbralDspHelpers::hannWindow(int size, bool periodic) {
    std::vector<Real> w(size);
    if (size <= 1) {
        std::fill(w.begin(), w.end(), 1.0f);
        return w;
    }
    const double denom = periodic ? static_cast<double>(size) : static_cast<double>(size - 1);
    for (int n = 0; n < size; ++n) {
        w[n] = static_cast<Real>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / denom));
    }
    return w;
}

static void computeRadix2Fft(std::vector<double>& real, std::vector<double>& imag, bool inverse = false) {
    const size_t N = real.size();
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    for (size_t len = 2; len <= N; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        for (size_t i = 0; i < N; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            for (size_t j = 0; j < len / 2; ++j) {
                const double u_re = real[i + j];
                const double u_im = imag[i + j];
                const double v_re = real[i + j + len / 2] * w_re - imag[i + j + len / 2] * w_im;
                const double v_im = real[i + j + len / 2] * w_im + imag[i + j + len / 2] * w_re;
                real[i + j] = u_re + v_re;
                imag[i + j] = u_im + v_im;
                real[i + j + len / 2] = u_re - v_re;
                imag[i + j + len / 2] = u_im - v_im;
                const double next_w_re = w_re * wlen_re - w_im * wlen_im;
                const double next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
    }
    if (inverse) {
        const double invN = 1.0 / static_cast<double>(N);
        for (size_t i = 0; i < N; ++i) {
            real[i] *= invN;
            imag[i] *= invN;
        }
    }
}

std::vector<Real> TimbralDspHelpers::powerSpectrum(const std::vector<Real>& frame, int nFft) {
    if (nFft <= 0) return {};
    std::vector<double> real(nFft, 0.0);
    std::vector<double> imag(nFft, 0.0);
    const size_t copyLen = std::min(frame.size(), static_cast<size_t>(nFft));
    for (size_t i = 0; i < copyLen; ++i) {
        real[i] = static_cast<double>(frame[i]);
    }
    computeRadix2Fft(real, imag);
    const size_t outBins = static_cast<size_t>(nFft / 2 + 1);
    std::vector<Real> p(outBins);
    for (size_t k = 0; k < outBins; ++k) {
        p[k] = static_cast<Real>(real[k] * real[k] + imag[k] * imag[k]);
    }
    return p;
}

std::vector<Real> TimbralDspHelpers::magnitudeSpectrum(const std::vector<Real>& frame, int nFft) {
    if (nFft <= 0) return {};
    std::vector<double> real(nFft, 0.0);
    std::vector<double> imag(nFft, 0.0);
    const size_t copyLen = std::min(frame.size(), static_cast<size_t>(nFft));
    for (size_t i = 0; i < copyLen; ++i) {
        real[i] = static_cast<double>(frame[i]);
    }
    computeRadix2Fft(real, imag);
    const size_t outBins = static_cast<size_t>(nFft / 2 + 1);
    std::vector<Real> m(outBins);
    for (size_t k = 0; k < outBins; ++k) {
        m[k] = static_cast<Real>(std::sqrt(real[k] * real[k] + imag[k] * imag[k]));
    }
    return m;
}

Real TimbralDspHelpers::spectralCentroid(const std::vector<Real>& powerSpec, Real sampleRate) {
    if (powerSpec.empty() || powerSpec.size() <= 1) {
        return 0.0f;
    }
    const size_t numBins = powerSpec.size();
    const double nyquist = static_cast<double>(sampleRate) / 2.0;
    const double binHz = nyquist / static_cast<double>(numBins - 1);
    double sumWeighted = 0.0;
    double sumPower = 0.0;
    for (size_t k = 0; k < numBins; ++k) {
        const double p = static_cast<double>(powerSpec[k]);
        sumWeighted += (static_cast<double>(k) * binHz) * p;
        sumPower += p;
    }
    if (sumPower <= 0.0) {
        return 0.0f;
    }
    return static_cast<Real>(sumWeighted / sumPower);
}

std::tuple<std::vector<int>, std::vector<Real>, std::vector<Real>>
TimbralDspHelpers::detectPeaksMIR(const std::vector<Real>& array,
                                 const std::vector<Real>& freqs,
                                 Real cthr,
                                 const std::vector<Real>& unprocessed) {
    if (array.empty() || freqs.size() != array.size()) {
        return {{}, {}, {}};
    }

    const size_t N = array.size();
    const std::vector<Real>& unproc = (!unprocessed.empty() && unprocessed.size() == N) ? unprocessed : array;

    std::vector<double> arrayAppended(N + 2);
    std::vector<double> unprocAppended(N + 2);
    std::vector<double> freqAppended(N + 2);

    arrayAppended[0] = -2.0;
    unprocAppended[0] = -2.0;
    freqAppended[0] = -1.0;

    for (size_t i = 0; i < N; ++i) {
        arrayAppended[i + 1] = static_cast<double>(array[i]);
        unprocAppended[i + 1] = static_cast<double>(unproc[i]);
        freqAppended[i + 1] = static_cast<double>(freqs[i]);
    }

    arrayAppended[N + 1] = -2.0;
    unprocAppended[N + 1] = -2.0;
    freqAppended[N + 1] = -1.0;

    std::vector<int> mx;
    for (size_t k = 0; k < N; ++k) {
        if (array[k] >= cthr) {
            double diffBefore = arrayAppended[k + 1] - arrayAppended[k];
            double diffAfter = arrayAppended[k + 2] - arrayAppended[k + 1];
            if (diffBefore > 0.0 && diffAfter <= 0.0) {
                mx.push_back(static_cast<int>(k + 1));
            }
        }
    }

    if (mx.empty()) {
        return {{}, {}, {}};
    }

    std::vector<int> finalmx;
    int j = 0;
    int mxj = mx[j];
    size_t jj = 1;
    double bufmin = 2.0;
    double bufmax = arrayAppended[mxj];
    double oldbufmin = arrayAppended[0];
    if (mxj > 1) {
        for (int k = 0; k < mxj - 1; ++k) {
            if (arrayAppended[k] < oldbufmin) oldbufmin = arrayAppended[k];
        }
    }

    const double cthrD = static_cast<double>(cthr);

    while (jj < mx.size()) {
        if (mx[jj - 1] + 1 == mx[jj] - 1) {
            bufmin = std::min(bufmin, arrayAppended[mx[jj - 1]]);
        } else {
            double segMin = arrayAppended[mx[jj - 1]];
            for (int k = mx[jj - 1]; k < mx[jj] - 1; ++k) {
                if (arrayAppended[k] < segMin) segMin = arrayAppended[k];
            }
            bufmin = std::min(bufmin, segMin);
        }

        if (bufmax - bufmin < cthrD) {
            if (arrayAppended[mx[jj]] > bufmax) {
                j = static_cast<int>(jj);
                mxj = mx[j];
                bufmax = arrayAppended[mxj];
                oldbufmin = std::min(oldbufmin, bufmin);
                bufmin = 2.0;
            } else if (arrayAppended[mx[jj]] - bufmax <= 0.0) {
                bufmax = std::max(bufmax, arrayAppended[mx[jj]]);
                oldbufmin = std::min(oldbufmin, bufmin);
            }
        } else {
            if (bufmax - oldbufmin < cthrD) {
                oldbufmin = std::min(oldbufmin, bufmin);
            } else {
                finalmx.push_back(mxj);
                oldbufmin = bufmin;
            }
            bufmax = arrayAppended[mx[jj]];
            j = static_cast<int>(jj);
            mxj = mx[j];
            bufmin = 2.0;
        }
        jj++;
    }

    double trailMin = arrayAppended[mx[j] + 1];
    for (size_t k = mx[j] + 1; k < arrayAppended.size(); ++k) {
        if (arrayAppended[k] < trailMin) trailMin = arrayAppended[k];
    }
    if ((bufmax - oldbufmin >= cthrD) && (bufmax - trailMin >= cthrD)) {
        finalmx.push_back(mx[j]);
    }

    std::stable_sort(finalmx.begin(), finalmx.end(), [&](int a, int b) {
        return arrayAppended[a] > arrayAppended[b];
    });

    std::vector<int> peakIdx(finalmx.size());
    std::vector<Real> peakLevel(finalmx.size());
    std::vector<Real> peakX(finalmx.size());

    for (size_t i = 0; i < finalmx.size(); ++i) {
        int m = finalmx[i];
        peakIdx[i] = m - 1;
        peakLevel[i] = static_cast<Real>(unprocAppended[m]);
        peakX[i] = static_cast<Real>(freqAppended[m]);
    }

    return {peakIdx, peakLevel, peakX};
}

Real TimbralDspHelpers::computeFrameRoughness(const std::vector<Real>& peakFreqs,
                                             const std::vector<Real>& peakLevels) {
    if (peakFreqs.size() <= 1 || peakFreqs.size() != peakLevels.size()) {
        return 0.0f;
    }
    double sumRough = 0.0;
    const size_t n = peakFreqs.size();
    for (size_t i = 1; i < n; ++i) {
        const double f1 = static_cast<double>(peakFreqs[i]);
        const double v1 = static_cast<double>(peakLevels[i]);
        for (size_t j = 0; j < i; ++j) {
            const double f2 = static_cast<double>(peakFreqs[j]);
            const double v2 = static_cast<double>(peakLevels[j]);

            const double min_f = std::min(f1, f2);
            const double s = 0.24 / (0.0207 * min_f + 18.96);
            const double delta_f = std::abs(f2 - f1);
            const double pd = std::exp(-3.51 * s * delta_f) - std::exp(-5.75 * s * delta_f);
            const double x_term = v1 * v2;
            const double y_term = (2.0 * v2) / (v1 + v2 + 1e-12);
            const double rough = std::pow(x_term, 0.1) * (0.5 * std::pow(y_term, 3.11)) * pd;
            sumRough += rough;
        }
    }
    return static_cast<Real>(sumRough);
}

Real TimbralDspHelpers::computeRoughness(const std::vector<Real>& audio,
                                       Real sampleRate,
                                       bool clipOutput) {
    if (audio.empty()) {
        return 0.0f;
    }
    Real maxAbs = 0.0f;
    for (Real s : audio) {
        Real a = std::abs(s);
        if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs == 0.0f) {
        return 0.0f;
    }

    const size_t padFront = 512;
    std::vector<Real> sig(audio.size() + padFront, 0.0f);
    std::copy(audio.begin(), audio.end(), sig.begin() + padFront);
    const size_t audioLen = sig.size();

    const double timeStep = 0.05;
    const int stepSamples = static_cast<int>(sampleRate * timeStep);
    const int nfft = stepSamples;

    std::vector<double> window(nfft);
    const double denomWin = static_cast<double>(nfft + 1);
    for (int n = 0; n < nfft; ++n) {
        window[n] = 0.54 - 0.46 * std::cos(2.0 * M_PI * static_cast<double>(n + 1) / denomWin);
    }

    const double olap = static_cast<double>(nfft) / 2.0;
    const int numFrames = static_cast<int>(static_cast<double>(audioLen) / (static_cast<double>(stepSamples) - olap));
    if (numFrames <= 0) {
        return 0.0f;
    }

    int nextPow2 = 1;
    while (nextPow2 <= stepSamples) {
        nextPow2 <<= 1;
    }

    const size_t specLen = static_cast<size_t>(nextPow2 / 2) + 1;
    std::vector<std::vector<Real>> spec(numFrames, std::vector<Real>(specLen, 0.0f));

    if (audioLen > static_cast<size_t>(stepSamples)) {
        size_t i = 0;
        size_t startIdx = 0;
        while (startIdx + static_cast<size_t>(stepSamples) <= audioLen && i < static_cast<size_t>(numFrames)) {
            std::vector<Real> frameBuf(nextPow2, 0.0f);
            for (int k = 0; k < stepSamples; ++k) {
                frameBuf[k] = static_cast<Real>(static_cast<double>(sig[startIdx + k]) * window[k]);
            }
            spec[i] = magnitudeSpectrum(frameBuf, nextPow2);
            i++;
            startIdx = static_cast<size_t>(static_cast<double>(i) * (static_cast<double>(nfft) / 2.0));
        }
    } else {
        std::vector<Real> frameBuf(nextPow2, 0.0f);
        for (size_t k = 0; k < audioLen; ++k) {
            frameBuf[k] = sig[k];
        }
        spec[0] = magnitudeSpectrum(frameBuf, nextPow2);
    }

    std::vector<Real> freq(specLen);
    for (size_t k = 0; k < specLen; ++k) {
        freq[k] = static_cast<Real>((sampleRate / 2.0f) * (static_cast<float>(k) / static_cast<float>(specLen - 1)));
    }

    Real minS = spec[0][0];
    Real maxS = spec[0][0];
    for (size_t fi = 0; fi < static_cast<size_t>(numFrames); ++fi) {
        for (size_t k = 0; k < specLen; ++k) {
            Real val = spec[fi][k];
            if (val < minS) minS = val;
            if (val > maxS) maxS = val;
        }
    }

    std::vector<std::vector<Real>> normSpec(numFrames, std::vector<Real>(specLen, 0.0f));
    if (maxS > minS) {
        Real range = maxS - minS;
        for (size_t fi = 0; fi < static_cast<size_t>(numFrames); ++fi) {
            for (size_t k = 0; k < specLen; ++k) {
                normSpec[fi][k] = (spec[fi][k] - minS) / range;
            }
        }
    }

    double sumAllRoughness = 0.0;
    int countFrames = 0;
    for (size_t fi = 0; fi < static_cast<size_t>(numFrames); ++fi) {
        auto [peakIdx, peakLevel, peakX] = detectPeaksMIR(normSpec[fi], freq, 0.01f, spec[fi]);
        if (peakX.size() > 1) {
            Real r = computeFrameRoughness(peakX, peakLevel);
            sumAllRoughness += static_cast<double>(r);
        }
        countFrames++;
    }

    double meanRoughness = (countFrames > 0) ? (sumAllRoughness / static_cast<double>(countFrames)) : 0.0;

    Real roughnessVal = 0.0f;
    if (meanRoughness >= 0.01) {
        roughnessVal = static_cast<Real>(std::log10(meanRoughness) * 13.98779569 + 48.97606571545886);
    }
    if (clipOutput) {
        roughnessVal = std::max(0.0f, std::min(100.0f, roughnessVal));
    }
    return roughnessVal;
}

Real TimbralDspHelpers::r2LinearFit(const std::vector<Real>& x, const std::vector<Real>& y) {
    const size_t n = x.size();
    if (n < 2 || n != y.size()) {
        return 0.0f;
    }
    double sumX = 0.0, sumY = 0.0, sumX2 = 0.0, sumXY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        sumX += xi;
        sumY += yi;
        sumX2 += xi * xi;
        sumXY += xi * yi;
    }
    const double denom = static_cast<double>(n) * sumX2 - sumX * sumX;
    if (denom == 0.0) {
        return 0.0f;
    }
    const double m = (static_cast<double>(n) * sumXY - sumX * sumY) / denom;
    const double c = (sumY - m * sumX) / static_cast<double>(n);
    const double meanY = sumY / static_cast<double>(n);

    double ssRes = 0.0;
    double ssTot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        const double yPred = m * xi + c;
        const double dRes = yi - yPred;
        const double dTot = yi - meanY;
        ssRes += dRes * dRes;
        ssTot += dTot * dTot;
    }
    if (ssTot <= 0.0) {
        return 0.0f;
    }
    const double r2 = 1.0 - (ssRes / ssTot);
    return static_cast<Real>(std::max(0.0, std::min(1.0, r2)));
}

Real TimbralDspHelpers::calculateHpssRatio(const std::vector<Real>& audio, Real sampleRate) {
    (void)sampleRate;
    if (audio.empty()) {
        return 1e-6f;
    }

    const size_t nFft = 2048;
    const size_t hop = 512;
    const size_t pad = nFft / 2;
    const size_t N = audio.size();

    // Pad front and back by pad
    std::vector<double> yPadded(N + 2 * pad, 0.0);
    for (size_t i = 0; i < N; ++i) {
        yPadded[pad + i] = static_cast<double>(audio[i]);
    }

    const size_t nFrames = (yPadded.size() - nFft) / hop + 1;
    const size_t numBins = nFft / 2 + 1;

    // Hann window
    std::vector<double> w(nFft);
    std::vector<double> wSq(nFft);
    for (size_t i = 0; i < nFft; ++i) {
        w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(nFft));
        wSq[i] = w[i] * w[i];
    }

    std::vector<std::vector<double>> D_re(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> D_im(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> S(numBins, std::vector<double>(nFrames));

    std::vector<double> frameReal(nFft);
    std::vector<double> frameImag(nFft);

    for (size_t t = 0; t < nFrames; ++t) {
        const size_t start = t * hop;
        for (size_t i = 0; i < nFft; ++i) {
            frameReal[i] = yPadded[start + i] * w[i];
            frameImag[i] = 0.0;
        }
        computeRadix2Fft(frameReal, frameImag, false);
        for (size_t k = 0; k < numBins; ++k) {
            D_re[k][t] = frameReal[k];
            D_im[k][t] = frameImag[k];
            S[k][t] = std::sqrt(frameReal[k] * frameReal[k] + frameImag[k] * frameImag[k]);
        }
    }

    // 2D median filtering
    std::vector<std::vector<double>> H(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> P(numBins, std::vector<double>(nFrames));

    // Time-direction filtering
    for (size_t k = 0; k < numBins; ++k) {
        for (size_t t = 0; t < nFrames; ++t) {
            std::array<double, 31> win;
            for (int p = -15; p <= 15; ++p) {
                int idx = static_cast<int>(t) + p;
                idx = std::max(0, std::min(static_cast<int>(nFrames) - 1, idx));
                win[p + 15] = S[k][idx];
            }
            std::nth_element(win.begin(), win.begin() + 15, win.end());
            H[k][t] = win[15];
        }
    }

    // Frequency-direction filtering
    for (size_t t = 0; t < nFrames; ++t) {
        for (size_t k = 0; k < numBins; ++k) {
            std::array<double, 31> win;
            for (int p = -15; p <= 15; ++p) {
                int idx = static_cast<int>(k) + p;
                idx = std::max(0, std::min(static_cast<int>(numBins) - 1, idx));
                win[p + 15] = S[idx][t];
            }
            std::nth_element(win.begin(), win.begin() + 15, win.end());
            P[k][t] = win[15];
        }
    }

    // Wiener soft masks
    std::vector<std::vector<double>> Dh_re(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> Dh_im(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> Dp_re(numBins, std::vector<double>(nFrames));
    std::vector<std::vector<double>> Dp_im(numBins, std::vector<double>(nFrames));

    for (size_t k = 0; k < numBins; ++k) {
        for (size_t t = 0; t < nFrames; ++t) {
            const double h2 = H[k][t] * H[k][t];
            const double p2 = P[k][t] * P[k][t];
            const double denom = h2 + p2 + 1e-12;
            const double maskH = h2 / denom;
            const double maskP = p2 / denom;
            Dh_re[k][t] = D_re[k][t] * maskH;
            Dh_im[k][t] = D_im[k][t] * maskH;
            Dp_re[k][t] = D_re[k][t] * maskP;
            Dp_im[k][t] = D_im[k][t] * maskP;
        }
    }

    // ISTFT reconstruction
    auto runIstft = [&](const std::vector<std::vector<double>>& D_real,
                        const std::vector<std::vector<double>>& D_imag) -> std::vector<double> {
        const size_t expectedLen = (nFrames - 1) * hop + nFft;
        std::vector<double> out(expectedLen, 0.0);
        std::vector<double> windowSum(expectedLen, 0.0);

        std::vector<double> fReal(nFft);
        std::vector<double> fImag(nFft);

        for (size_t t = 0; t < nFrames; ++t) {
            fReal[0] = D_real[0][t];
            fImag[0] = 0.0;
            fReal[nFft / 2] = D_real[numBins - 1][t];
            fImag[nFft / 2] = 0.0;
            for (size_t k = 1; k < numBins - 1; ++k) {
                fReal[k] = D_real[k][t];
                fImag[k] = D_imag[k][t];
                fReal[nFft - k] = D_real[k][t];
                fImag[nFft - k] = -D_imag[k][t];
            }

            computeRadix2Fft(fReal, fImag, true);

            const size_t start = t * hop;
            for (size_t i = 0; i < nFft; ++i) {
                out[start + i] += fReal[i] * w[i];
                windowSum[start + i] += wSq[i];
            }
        }

        for (size_t i = 0; i < expectedLen; ++i) {
            if (windowSum[i] > 1e-11) {
                out[i] /= windowSum[i];
            }
        }

        std::vector<double> res(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            if (pad + i < expectedLen) {
                res[i] = out[pad + i];
            }
        }
        return res;
    };

    std::vector<double> yh = runIstft(Dh_re, Dh_im);
    std::vector<double> yp = runIstft(Dp_re, Dp_im);

    // Frame RMS energy integration
    const size_t stepSize = 1024;
    const size_t overlapStep = 512;
    std::vector<double> ep;
    std::vector<double> eh;

    size_t i = 0;
    while (i + stepSize <= N) {
        double sumSqP = 0.0;
        double sumSqH = 0.0;
        for (size_t k = 0; k < stepSize; ++k) {
            sumSqP += yp[i + k] * yp[i + k];
            sumSqH += yh[i + k] * yh[i + k];
        }
        ep.push_back(std::sqrt(sumSqP / static_cast<double>(stepSize)));
        eh.push_back(std::sqrt(sumSqH / static_cast<double>(stepSize)));
        i += overlapStep;
    }

    if (i < N) {
        const size_t rem = N - i;
        double sumSqP = 0.0;
        double sumSqH = 0.0;
        for (size_t k = 0; k < rem; ++k) {
            sumSqP += yp[i + k] * yp[i + k];
            sumSqH += yh[i + k] * yh[i + k];
        }
        ep.push_back(std::sqrt(sumSqP / static_cast<double>(rem)));
        eh.push_back(std::sqrt(sumSqH / static_cast<double>(rem)));
    }

    double sumEp = 0.0;
    double sumEh = 0.0;
    for (size_t k = 0; k < ep.size(); ++k) {
        sumEp += ep[k];
        sumEh += eh[k];
    }

    const double total = sumEp + sumEh;
    if (total > 0.0) {
        return static_cast<Real>(sumEp / total);
    }
    return 1e-6f;
}

std::vector<Real> TimbralDspHelpers::calculateBandwidthArray(const std::vector<Real>& audio,
                                                           Real sampleRate,
                                                           int nperseg,
                                                           int hop,
                                                           Real thresholdDb) {
    if (audio.size() < static_cast<size_t>(nperseg) || hop <= 0 || nperseg <= 0) {
        return {};
    }

    const int nFrames = 1 + static_cast<int>(audio.size() - nperseg) / hop;
    if (nFrames <= 0) {
        return {};
    }

    const size_t numBins = static_cast<size_t>(nperseg / 2 + 1);
    const double scale = 1.0 / std::sqrt(static_cast<double>(sampleRate) * static_cast<double>(nperseg));

    std::vector<std::vector<double>> spec(numBins, std::vector<double>(nFrames, 0.0));
    std::vector<double> levelWithTime(nFrames, 0.0);

    std::vector<double> real(nperseg, 0.0);
    std::vector<double> imag(nperseg, 0.0);

    for (int frame = 0; frame < nFrames; ++frame) {
        const size_t start = static_cast<size_t>(frame * hop);
        double sum = 0.0;
        for (int i = 0; i < nperseg; ++i) {
            sum += static_cast<double>(audio[start + i]);
        }
        const double mean = sum / static_cast<double>(nperseg);

        for (int i = 0; i < nperseg; ++i) {
            real[i] = static_cast<double>(audio[start + i]) - mean;
            imag[i] = 0.0;
        }

        computeRadix2Fft(real, imag, false);

        double frameLevel = 0.0;
        for (size_t k = 0; k < numBins; ++k) {
            const double mag = std::sqrt(real[k] * real[k] + imag[k] * imag[k]) * scale;
            spec[k][frame] = mag;
            frameLevel += mag;
        }
        levelWithTime[frame] = frameLevel;
    }

    std::vector<double> f(numBins);
    const double nyquist = static_cast<double>(sampleRate) / 2.0;
    for (size_t k = 0; k < numBins; ++k) {
        f[k] = nyquist * (static_cast<double>(k) / static_cast<double>(numBins - 1));
    }

    double maxL = levelWithTime[0];
    double minL = levelWithTime[0];
    for (int frame = 1; frame < nFrames; ++frame) {
        if (levelWithTime[frame] > maxL) maxL = levelWithTime[frame];
        if (levelWithTime[frame] < minL) minL = levelWithTime[frame];
    }
    const double minTpower = (0.1 * (maxL - minL)) + minL;
    const double rolloffThresh = std::pow(10.0, static_cast<double>(thresholdDb) / 20.0);

    std::vector<Real> bandwidth(nFrames, 0.0f);

    for (int frame = 0; frame < nFrames; ++frame) {
        const double tpower = levelWithTime[frame];
        if (tpower > minTpower) {
            double cum = 0.0;
            size_t rollonBin = 0;
            const double rollonTarget = tpower * 0.05;
            for (size_t k = 0; k < numBins; ++k) {
                cum += spec[k][frame];
                if (cum >= rollonTarget) {
                    rollonBin = k;
                    break;
                }
            }

            int lastIdx = -1;
            for (int k = static_cast<int>(numBins) - 1; k >= 0; --k) {
                if (spec[k][frame] >= rolloffThresh) {
                    lastIdx = k;
                    break;
                }
            }

            if (lastIdx >= 0) {
                bandwidth[frame] = static_cast<Real>(f[lastIdx] - f[rollonBin]);
            } else {
                bandwidth[frame] = 0.0f;
            }
        } else {
            bandwidth[frame] = 0.0f;
        }
    }

    return bandwidth;
}

std::tuple<Real, Real, int>
TimbralDspHelpers::calculateAttackTime(const std::vector<Real>& envelopeSamples,
                                      Real sampleRate,
                                      Real maxAttackTime,
                                      const std::string& calculationType) {
    if (envelopeSamples.empty()) {
        const Real defLogAt = std::log10(1.0f / sampleRate);
        return {defLogAt, 0.0f, 0};
    }

    std::vector<double> env(envelopeSamples.size());
    double maxVal = 0.0;
    for (size_t i = 0; i < envelopeSamples.size(); ++i) {
        env[i] = static_cast<double>(envelopeSamples[i]);
        if (env[i] > maxVal) maxVal = env[i];
    }

    if (maxVal > 0.0) {
        for (size_t i = 0; i < env.size(); ++i) {
            env[i] /= maxVal;
        }
    }

    size_t peakIdx = 0;
    double maxEnv = env[0];
    for (size_t i = 1; i < env.size(); ++i) {
        if (env[i] > maxEnv) {
            maxEnv = env[i];
            peakIdx = i;
        }
    }

    if (peakIdx == 0) {
        const Real defLogAt = std::log10(1.0f / sampleRate);
        return {defLogAt, 0.0f, 0};
    }

    double minPrePeakVal = env[0];
    for (size_t i = 1; i < peakIdx; ++i) {
        if (env[i] < minPrePeakVal) {
            minPrePeakVal = env[i];
        }
    }
    size_t minPrePeakIdx = 0;
    for (size_t i = 0; i < peakIdx; ++i) {
        if (std::abs(env[i] - minPrePeakVal) < 1e-12) {
            minPrePeakIdx = i;
        }
    }

    const size_t segLen = peakIdx - minPrePeakIdx + 1;
    std::vector<double> seg(segLen);
    for (size_t i = 0; i < segLen; ++i) {
        seg[i] = env[minPrePeakIdx + i];
    }

    double segMin = seg[0];
    double segMax = seg[0];
    for (size_t i = 1; i < segLen; ++i) {
        if (seg[i] < segMin) segMin = seg[i];
        if (seg[i] > segMax) segMax = seg[i];
    }
    const double dynRange = segMax - segMin;

    if (calculationType == "min_effort") {
        const int threshNo = 8;
        const double m = 3.0;
        const double thresholdStep = 1.0 / (threshNo + 2);
        std::vector<double> threshLevel(threshNo + 1);
        for (int j = 0; j <= threshNo; ++j) {
            double frac = thresholdStep + (static_cast<double>(j) / static_cast<double>(threshNo)) * (1.0 - 2.0 * thresholdStep);
            threshLevel[j] = (frac * dynRange) + segMin;
        }

        std::vector<int> thresholdIdxs(threshNo + 1, 0);
        for (int j = 0; j <= threshNo; ++j) {
            for (size_t k = 0; k < segLen; ++k) {
                if (seg[k] >= threshLevel[j]) {
                    thresholdIdxs[j] = static_cast<int>(k);
                    break;
                }
            }
        }

        std::vector<double> effort(threshNo);
        double effortSum = 0.0;
        for (int j = 0; j < threshNo; ++j) {
            effort[j] = static_cast<double>(thresholdIdxs[j + 1] - thresholdIdxs[j]);
            effortSum += effort[j];
        }
        const double effortMean = effortSum / static_cast<double>(threshNo);
        const double effortThreshold = effortMean * m;

        int thStart = 0;
        for (int j = 0; j < threshNo; ++j) {
            if (effort[j] <= effortThreshold) {
                thStart = j;
                break;
            }
        }

        int thEndRel = (threshNo - thStart - 1);
        for (int j = thStart; j < threshNo; ++j) {
            if (effort[j] >= effortThreshold) {
                thEndRel = j - thStart;
                break;
            }
        }
        int thEnd = thStart + thEndRel;

        int thStartIdx = thresholdIdxs[thStart];
        int thEndIdx = thresholdIdxs[thEnd];

        if (thStartIdx == thEndIdx) {
            thStartIdx = thresholdIdxs[0];
            thEndIdx = thresholdIdxs[threshNo];
        }

        double attackTime = 0.0;
        if (thStartIdx == thEndIdx) {
            attackTime = 1.0 / static_cast<double>(sampleRate);
        } else {
            attackTime = static_cast<double>(thEndIdx - thStartIdx + 1) / static_cast<double>(sampleRate);
        }

        if (maxAttackTime > 0.0f && attackTime > static_cast<double>(maxAttackTime)) {
            const int maxAttackTimeSample = static_cast<int>(static_cast<double>(sampleRate) * static_cast<double>(maxAttackTime));
            thEndIdx = thStartIdx + maxAttackTimeSample;
            attackTime = static_cast<double>(thEndIdx - thStartIdx + 1) / static_cast<double>(sampleRate);
        }

        return {static_cast<Real>(std::log10(attackTime)), 0.0f, static_cast<int>(thStartIdx + minPrePeakIdx)};

    } else if (calculationType == "fixed_threshold") {
        const double lowerThreshold = (20.0 * dynRange * 0.01) + segMin;
        const double upperThreshold = (90.0 * dynRange * 0.01) + segMin;

        int thStartIdx = 0;
        for (size_t k = 0; k < segLen; ++k) {
            if (seg[k] >= lowerThreshold) {
                thStartIdx = static_cast<int>(k);
                break;
            }
        }

        int thEndIdx = thStartIdx;
        for (size_t k = static_cast<size_t>(thStartIdx); k < segLen; ++k) {
            if (seg[k] >= upperThreshold) {
                thEndIdx = static_cast<int>(k);
                break;
            }
        }

        double attackTime = 0.0;
        if (thStartIdx == thEndIdx) {
            attackTime = 1.0 / static_cast<double>(sampleRate);
        } else {
            attackTime = static_cast<double>(thEndIdx - thStartIdx + 1) / static_cast<double>(sampleRate);
        }

        if (maxAttackTime > 0.0f && attackTime > static_cast<double>(maxAttackTime)) {
            const int maxAttackTimeSample = static_cast<int>(static_cast<double>(sampleRate) * static_cast<double>(maxAttackTime));
            thEndIdx = thStartIdx + maxAttackTimeSample;
            attackTime = static_cast<double>(thEndIdx - thStartIdx + 1) / static_cast<double>(sampleRate);
        }

        return {static_cast<Real>(std::log10(attackTime)), 0.0f, static_cast<int>(thStartIdx + minPrePeakIdx)};
    }

    const Real defLogAt = std::log10(1.0f / sampleRate);
    return {defLogAt, 0.0f, 0};
}

Real TimbralDspHelpers::calculateAttackCentroid(const std::vector<Real>& audioSeg,
                                              Real sampleRate) {
    if (audioSeg.size() <= 512) {
        return 200.0f;
    }

    int nFft = 1;
    while (nFft < static_cast<int>(audioSeg.size())) {
        nFft <<= 1;
    }

    auto spectrum = magnitudeSpectrum(audioSeg, nFft);
    const size_t numBins = spectrum.size();
    const double nyquist = static_cast<double>(sampleRate) / 2.0;

    size_t lfStart = numBins;
    for (size_t k = 0; k < numBins; ++k) {
        double freqK = nyquist * (static_cast<double>(k) / static_cast<double>(numBins - 1));
        if (freqK >= 20.0) {
            lfStart = k;
            break;
        }
    }

    if (lfStart >= numBins) {
        return 200.0f;
    }

    double sumWeighted = 0.0;
    double tpower = 0.0;
    for (size_t k = lfStart; k < numBins; ++k) {
        double freqK = nyquist * (static_cast<double>(k) / static_cast<double>(numBins - 1));
        double specK = static_cast<double>(spectrum[k]);
        sumWeighted += specK * freqK;
        tpower += specK;
    }

    if (tpower > 0.0) {
        return static_cast<Real>(sumWeighted / tpower);
    }

    return 200.0f;
}

} // namespace essentia
