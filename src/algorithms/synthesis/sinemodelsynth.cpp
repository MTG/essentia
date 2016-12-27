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

#include "sinemodelsynth.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;


const char* SineModelSynth::name = "SineModelSynth";
const char* SineModelSynth::category = "Synthesis";
const char* SineModelSynth::description = DOC("This algorithm computes the sine model synthesis from sine model analysis.");

void SineModelSynth::compute() {

  const std::vector<Real>& magnitudes = _magnitudes.get();
  const std::vector<Real>& frequencies = _frequencies.get();
  const std::vector<Real>& phases = _phases.get();

  std::vector<std::complex<Real> >& outfft = _outfft.get();

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

