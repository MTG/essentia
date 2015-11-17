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

#include "sinesubtraction.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;

/*
	"""
	Subtract sinusoids from a sound
	x: input sound, N: fft-size, H: hop-size
	sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
	returns xr: residual sound
	"""

	hN = N/2                                           # half of fft size
	x = np.append(np.zeros(hN),x)                      # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hN))                      # add zeros at the end to analyze last sample
	bh = blackmanharris(N)                             # blackman harris window
	w = bh/ sum(bh)                                    # normalize window
	sw = np.zeros(N)                                   # initialize synthesis window
	sw[hN-H:hN+H] = triang(2*H) / w[hN-H:hN+H]         # synthesis window
	L = sfreq.shape[0]                                 # number of frames, this works if no sines
	xr = np.zeros(x.size)                              # initialize output array
	pin = 0
	# jjaner debug
	outresmag = []
	for l in range(L):
		xw = x[pin:pin+N]*w                              # window the input sound
		X = fft(fftshift(xw))                            # compute FFT
		Yh = UF_C.genSpecSines(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines
		Xr = X-Yh
		print Xr.shape                                     # subtract sines from original spectrum
		outresmag = np.append(outresmag, abs(Xr))
		xrw = np.real(fftshift(ifft(Xr)))                # inverse FFT
		xr[pin:pin+N] += xrw*sw                          # overlap-add
		pin += H                                         # advance sound pointer
	xr = np.delete(xr, range(hN))                      # delete half of first window which was added in stftAnal
	xr = np.delete(xr, range(xr.size-hN, xr.size))     # delete half of last window which was added in stftAnal

	print("jj debug: writing output residual magnitude")
	np.savetxt('outresmag.txt',outresmag)
	return xr
	*/

const char* SineSubtraction::name = "SineSubtraction";
const char* SineSubtraction::description = DOC("This algorithm subtracts the sinusoids computed with the sine model analysis from an input audio signal. It ouputs an audio signal.");


void SineSubtraction::configure() {

    _sampleRate = parameter("sampleRate").toReal();
    _fftSize = parameter("fftSize").toInt();
    _hopSize = parameter("hopSize").toInt();

	 	std::string wtype = "blackmanharris92"; // default "hamming"
		_window->configure( "type", wtype.c_str());

	// create synthesis window
	createSynthesisWindow(_synwindow, _fftSize);


  }
}

void SineSubtraction::compute() {

  const std::vector<Real>& inframe = _inframe.get();
  const std::vector<Real>& magnitudes = _magnitudes.get();
  const std::vector<Real>& frequencies = _frequencies.get();
  const std::vector<Real>& phases = _phases.get();


  std::vector<Real>& outframe = _outframe.get();

  std::vector<Real> sinesframe;

	// compute input frame FFT
	std::vector<Real> synframe;
	std::vector<Real> wsynframe;
	std::vector<std::complex<Real> > synfft;
	for (int i= (inframe.size()/2) - _fftSize/2; i <=  (inframe.size()/2) + _fftSize/2; ++i)
	{
		synframe.push_back(inframe[i]);
	}
	printf("size synframe %d, ", sinframe.size());

	_window->input("frame").set(synframe);
	_window->output("frame").set(wsynframe);

	fft->input("frame").set(wsynframe);
	fft->output("fft").set(synfft);
	fft->compute();

	// generate sine spectrum
	std::vector<std::complex<Real> > sinefft;
	generateSines(magnitudes, frequencies, phases, sinefft);


  // subtract  sines in FFT domain
	subtractFFT(synfft, sinefft);

  // IFFT of subtracted spectra
	ifft->input("fft").set(synfft);
	ifft->output("frame").set(synframeout);
	ifft->compute();
	// overlapp add synthesized audio
	overlapadd->input("frame").set(synframeout);
	overlapadd->output("frame").set(outframe);
	overlapadd->compute();

}

void 	SineSubtraction::subtractFFT(std::vector<std::complex<Real> >&fft1, const std::vector<std::complex<Real> >&fft2)
{
  int minSize = std::min((int)fft1.size(), (int)fft2.size());
  for (int i=0; i < minSize; ++i){
    fft1[i].real( fft1[i].real() -  fft2[i].real());
    fft1[i].imag( fft1[i].imag() -  fft2[i].imag());
  }
}



void SineSubtraction::initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT)
{
  fft.resize(sizeFFT);
  for (int i=0; i < sizeFFT; ++i){
    fft[i].real(0);
    fft[i].imag(0);
  }
}



void Sinesubtraction::generateSines(const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<std::complex<Real> >&outfft)
{
	int outSize = (int)floor(_fftSize/2.0) + 1;
  initializeFFT(outfft, outSize);
  int i = 0;

  // convert frequencies to peak locations
  std::vector<Real> locs(frequencies.size());
  for (i=0; i < int(frequencies.size()); ++i){
    locs[i] = _fftSize*frequencies[i]/float(_sampleRate);
  }
  // init synth phase vector
  std::vector<Real> ytphase(frequencies.size());
  std::fill(ytphase.begin(), ytphase.end(), 0.);

  // initialize last phase and frequency vectors
  if (_lastytphase.size() < ytphase.size())
  {
    _lastytphase.resize(ytphase.size());
    std::fill(_lastytphase.begin(), _lastytphase.end(), 0.);
  }
  if (_lastytfreq.size() < frequencies.size())
  {
    _lastytfreq.resize(frequencies.size());
    std::fill(_lastytfreq.begin(), _lastytfreq.end(), 0.);
  }


  // propagate phase if necessary (no input phase vector)
  if (int(phases.size()) > 0){                                 // if no phases generate them
	  	ytphase = phases;
	  }
  else{
		for (i=0; i < int(ytphase.size()); ++i)
		{
			ytphase[i] = _lastytphase[i] + (M_PI * (_lastytfreq[i] + frequencies[i])/float(_sampleRate)) * _hopSize;     // propagate phases
    }
  }

  // generate output fft
  genSpecSines(locs, magnitudes, ytphase, outfft, _fftSize);

  for (i = 0; i < int(ytphase.size()); ++i)
  {
		ytphase[i] = fmod (ytphase[i], float(2*M_PI));                        // make phase inside 2*pi
  }

  // save frequency and phase for phase propagation
  _lastytfreq = frequencies;
  _lastytphase = ytphase;


}

void SineSubtraction::createSynthesisWindow(std::vector<Real> &synwindow, int hopSize, int winSize)
{
std::vector<Real> ones;
std::vector<Real> triangle;
std::vector<Real> win;

for (int i=0; i < winSize;++i){
	ones.push_back(1.f);
}

_window->input.set(ones);
_window->output.set(win);
_window->compute();

// create traingular
std::string wtype = "triangular"; // default "hamming"
Algorithm* trinagular = factory.create("Windowing", "type", wtype.c_str());
ones.reisze(2*hopSize);
triangular->input.set(ones);
triangular->output.set(triangle);
triangular->compute();

// init synthesis window
synwindow.resize(winSize);
std::fill(synwindow.begin(), synwindow.end(), 0.);
int hN = winSize / 2;
for (int i= hN-hopSize; i < hN+ hopSize;++i){
	synwindow[i] = triangle[i-(hN-hopSize)] / win[i];
}



delete triangular;

}


