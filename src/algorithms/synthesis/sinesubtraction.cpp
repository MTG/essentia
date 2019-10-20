/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "sinesubtraction.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;


const char* SineSubtraction::name = "SineSubtraction";
const char* SineSubtraction::category = "Synthesis";
const char* SineSubtraction::description = DOC("This algorithm subtracts the sinusoids computed with the sine model analysis from an input audio signal. It ouputs an audio signal.");


void SineSubtraction::configure() {

    _sampleRate = parameter("sampleRate").toReal();
    _fftSize = parameter("fftSize").toInt();
    _hopSize = parameter("hopSize").toInt();
    
      // configure algorithms
          std::string wtype = "blackmanharris92"; // default "hamming"
      _window->configure( "type", wtype.c_str());

    _fft->configure("size", _fftSize);

    _overlapadd->configure("frameSize", _fftSize, // uses synthesis window
                           "hopSize", _hopSize);
    // create synthesis window
    createSynthesisWindow(_synwindow, _hopSize, _fftSize);

}

void SineSubtraction::compute() {

  const std::vector<Real>& inframe = _inframe.get();
  const std::vector<Real>& magnitudes = _magnitudes.get();
  const std::vector<Real>& frequencies = _frequencies.get();
  const std::vector<Real>& phases = _phases.get();

  std::vector<Real>& outframe = _outframe.get();
//
  std::vector<Real> sinesframe;

    // compute input frame FFT
    std::vector<Real> synframe;
    std::vector<Real> wsynframe;
    std::vector<Real> synframeout;

    std::vector<std::complex<Real> > synfft;
    for (int i= (int) ((inframe.size()/2) - _fftSize/2); i <  (int) ((inframe.size()/2) + _fftSize/2); ++i) {
        synframe.push_back(inframe[i]);
    }

    _window->input("frame").set(synframe);
    _window->output("frame").set(wsynframe);
    _window->compute();

    _fft->input("frame").set(wsynframe);
    _fft->output("fft").set(synfft);
    _fft->compute();

    // generate sine spectrum
    std::vector<std::complex<Real> > sinefft;
    generateSines(magnitudes, frequencies, phases, sinefft);

  // subtract  sines in FFT domain
    subtractFFT(synfft, sinefft);

  // IFFT of subtracted spectra
    _ifft->input("fft").set(synfft);
    _ifft->output("frame").set(synframeout);
    _ifft->compute();

    applySynthesisWindow(synframeout, _synwindow);

    // overlapp add synthesized audio
    _overlapadd->input("signal").set(synframeout);
    _overlapadd->output("signal").set(outframe);
    _overlapadd->compute();

}

void 	SineSubtraction::subtractFFT(std::vector<std::complex<Real> >&fft1, const std::vector<std::complex<Real> >&fft2) {
  int minSize = std::min((int)fft1.size(), (int)fft2.size());
  for (int i=0; i < minSize; ++i) {
    fft1[i].real( fft1[i].real() -  fft2[i].real());
    fft1[i].imag( fft1[i].imag() -  fft2[i].imag());
  }
}



void SineSubtraction::initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT) {
  fft.resize(sizeFFT);
  for (int i=0; i < sizeFFT; ++i){
    fft[i].real(0);
    fft[i].imag(0);
  }
}



void SineSubtraction::generateSines(const std::vector<Real> magnitudes,
                                    const std::vector<Real> frequencies,
                                    const std::vector<Real> phases,
                                    std::vector<std::complex<Real> >&outfft) {
  int outSize = (int)floor(_fftSize/2.0) + 1;

  initializeFFT(outfft, outSize);
  int i = 0;

  // convert frequencies to peak locations
  std::vector<Real> locs(frequencies.size());
  for (i=0; i < int(frequencies.size()); ++i) {
    locs[i] = _fftSize*frequencies[i]/float(_sampleRate);
  }
  // init synth phase vector
  std::vector<Real> ytphase(frequencies.size());
  std::fill(ytphase.begin(), ytphase.end(), 0.);

  // initialize last phase and frequency vectors
  if (_lastytphase.size() < ytphase.size()) {
    _lastytphase.resize(ytphase.size());
    std::fill(_lastytphase.begin(), _lastytphase.end(), 0.);
  }
  if (_lastytfreq.size() < frequencies.size()) {
    _lastytfreq.resize(frequencies.size());
    std::fill(_lastytfreq.begin(), _lastytfreq.end(), 0.);
  }

  // propagate phase if necessary (no input phase vector)
  if (int(phases.size()) > 0) {  // if no phases generate them
    ytphase = phases;
  } else {
      for (i=0; i < int(ytphase.size()); ++i) {
        ytphase[i] = _lastytphase[i] + (M_PI * (_lastytfreq[i] + frequencies[i]) /
                                                float(_sampleRate)) * _hopSize;     // propagate phases
      }
  }

  // generate output fft
  genSpecSines(locs, magnitudes, ytphase, outfft, _fftSize);

  for (i = 0; i < int(ytphase.size()); ++i) {
    ytphase[i] = fmod (ytphase[i], float(2*M_PI));  // make phase inside 2*pi
  }

  // save frequency and phase for phase propagation
  _lastytfreq = frequencies;
  _lastytphase = ytphase;


}

void SineSubtraction::createSynthesisWindow(std::vector<Real> &synwindow, int hopSize, int winSize) {
    std::vector<Real> ones;
    std::vector<Real> triangle;
    std::vector<Real> win;

    for (int i=0; i < winSize;++i) {
        ones.push_back(1.f);
    }

    _window->input("frame").set(ones);
    _window->output("frame").set(win);
    _window->compute();

    // create traingular
    Algorithm* triangular;
    std::string wtype = "triangular"; // default "hamming"
    triangular = AlgorithmFactory::create("Windowing", "type", wtype.c_str());
    ones.resize(2*hopSize); // trim to size 2*hopsize
    triangular->input("frame").set(ones);
    triangular->output("frame").set(triangle);
    triangular->compute();

    // init synthesis window
    synwindow.resize(winSize);
    std::fill(synwindow.begin(), synwindow.end(), 0.);

    //int hN = winSize / 2;

    // first half of the windowed signal is the
    // second half of the signal with windowing!
    int i=0;
    for (int j=0; j< hopSize; j++) {
        synwindow[i++] = triangle[j] / win[j];
    }

    // second half of the signal
    i = winSize - hopSize;
    for (int j= hopSize; j< 2 * hopSize; j++) {
        synwindow[i] = triangle[j] / win[i];
        i++;
    }

    delete triangular;

}

void SineSubtraction::applySynthesisWindow(std::vector<Real> &inframe, const std::vector<Real> synwindow) {
// it considers already the zero-phase window shift
    int signalSize = (int)inframe.size();

    for (int i= 0 ; i < signalSize ;++i){
      inframe[i] *= synwindow[i];
    }

}
