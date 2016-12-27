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
 * You should ha ve received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "sprmodelanal.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;

const char* SprModelAnal::name = "SprModelAnal";
const char* SprModelAnal::category = "Synthesis";
const char* SprModelAnal::description = DOC("This algorithm computes the sinusoidal plus residual model analysis. \n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. This algorithm uses SineModelAnal. See documentation for possible exceptions and input requirements on input \"spectrum\".\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);



void SprModelAnal::configure() {

  std::string wtype = "blackmanharris92"; // default "hamming"
  _window->configure("type", wtype.c_str());

  _fft->configure("size", parameter("fftSize").toInt()  );


  _sineModelAnal->configure( "sampleRate", parameter("sampleRate").toReal(),
                              "maxnSines", parameter("maxnSines").toInt() ,
                              "freqDevOffset", parameter("freqDevOffset").toInt(),
                              "freqDevSlope",  parameter("freqDevSlope").toReal()
                              );

  int subtrFFTSize = std::min(parameter("fftSize").toInt()/4, 4*parameter("hopSize").toInt());  // make sure the FFT size i
  _sineSubtraction->configure( "sampleRate", parameter("sampleRate").toReal(),
                              "fftSize", subtrFFTSize,
                              "hopSize", parameter("hopSize").toInt()
                              );


}


void SprModelAnal::compute() {

  // inputs and outputs
  const std::vector<Real>& frame = _frame.get();

  std::vector<Real>& peakMagnitude = _magnitudes.get();
  std::vector<Real>& peakFrequency = _frequencies.get();
  std::vector<Real>& peakPhase = _phases.get();
  std::vector<Real>& res = _res.get();

  std::vector<Real> wframe;
  std::vector<std::complex<Real> > fftin;
  std::vector<Real> fftmag;
  std::vector<Real> fftphase;


  _window->input("frame").set(frame);
  _window->output("frame").set(wframe);
  _window->compute();

  _fft->input("frame").set(wframe);
  _fft->output("fft").set(fftin);
  _fft->compute();

 _sineModelAnal->input("fft").set(fftin);
 _sineModelAnal->output("magnitudes").set(peakMagnitude);
 _sineModelAnal->output("frequencies").set(peakFrequency);
 _sineModelAnal->output("phases").set(peakPhase);

  _sineModelAnal->compute();


  std::vector<Real> subtrFrameOut;

// this needs to take into account overlap-add issues, introducing delay
 _sineSubtraction->input("frame").set(frame); // size is iput _fftSize
 _sineSubtraction->input("magnitudes").set(peakMagnitude);
 _sineSubtraction->input("frequencies").set(peakFrequency);
 _sineSubtraction->input("phases").set(peakPhase);
 _sineSubtraction->output("frame").set(res); // Nsyn size (subtrFFTSize)
 _sineSubtraction->compute();



}



