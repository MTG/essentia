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

#include "hprmodelanal.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;

const char* HprModelAnal::name = "HprModelAnal";
const char* HprModelAnal::category = "Synthesis";
const char* HprModelAnal::description = DOC("This algorithm computes the harmonic plus residual model analysis.\n"
"\n"
"It uses the algorithms HarmonicModelAnal and SineSubtraction .\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);



void HprModelAnal::configure() {

  std::string wtype = "blackmanharris92"; // default "hamming"

  _window->configure("type", wtype.c_str());

  _fft->configure("size", parameter("fftSize").toInt()  );

  _harmonicModelAnal->configure( "sampleRate", parameter("sampleRate").toReal(),                                                           
                              "hopSize", parameter("hopSize").toInt(),                     
                              "maxnSines", parameter("maxnSines").toInt() ,
                              "freqDevOffset", parameter("freqDevOffset").toReal(),
                              "freqDevSlope",  parameter("freqDevSlope").toReal(),
                              "nHarmonics",   parameter("nHarmonics").toInt(),                     
                              "harmDevSlope",   parameter("harmDevSlope").toReal(),
                              "maxFrequency",  parameter("maxFrequency").toReal(),
                              "minFrequency",  parameter("minFrequency").toReal()
                              
);

  int subtrFFTSize = std::min(512, 4*parameter("hopSize").toInt());  // make sure the FFT size is at least twice the hopsize
  
  _sineSubtraction->configure( "sampleRate", parameter("sampleRate").toReal(),
                              "fftSize", subtrFFTSize,
                              "hopSize", parameter("hopSize").toInt()
                              );

}


void HprModelAnal::compute() {

  // inputs and outputs
  const std::vector<Real>& frame = _frame.get();
  const Real & pitch = _pitch.get();
  
  std::vector<Real>& peakMagnitude = _magnitudes.get();
  std::vector<Real>& peakFrequency = _frequencies.get();
  std::vector<Real>& peakPhase = _phases.get();
  std::vector<Real>& res = _res.get();

  std::vector<Real> wframe;
  std::vector<std::complex<Real> > fftin;
  
  _window->input("frame").set(frame);
  _window->output("frame").set(wframe);
  _window->compute();
  
  
  _fft->input("frame").set(wframe);
  _fft->output("fft").set(fftin);
  _fft->compute();
  
 _harmonicModelAnal->input("fft").set(fftin);
 _harmonicModelAnal->input("pitch").set(pitch);
 _harmonicModelAnal->output("magnitudes").set(peakMagnitude);
 _harmonicModelAnal->output("frequencies").set(peakFrequency);
 _harmonicModelAnal->output("phases").set(peakPhase);

  _harmonicModelAnal->compute();

// this needs to take into account overlap-add issues, introducing delay
 _sineSubtraction->input("frame").set(frame); // size is iput _fftSize
 _sineSubtraction->input("magnitudes").set(peakMagnitude);
 _sineSubtraction->input("frequencies").set(peakFrequency);
 _sineSubtraction->input("phases").set(peakPhase);
 _sineSubtraction->output("frame").set(res); // Nsyn size
 _sineSubtraction->compute();


}




