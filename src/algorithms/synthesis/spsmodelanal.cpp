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

#include "spsmodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SpsModelAnal::name = "SpsModelAnal";
const char* SpsModelAnal::description = DOC("This algorithm computes the stochastic model analysis. \n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. This algorithm uses SineModelAnal. See documentation for possible exceptions and input requirements on input \"spectrum\".\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);




void SpsModelAnal::configure() {

//  std::string orderBy = parameter("orderBy").toLower();
//  if (orderBy == "magnitude") {
//    orderBy = "amplitude";
//  }
//  else if (orderBy == "frequency") {
//    orderBy = "position";
//  }
//  else {
//    throw EssentiaException("Unsupported ordering type: '" + orderBy + "'");
//  }

//  _peakDetect->configure("interpolate", true,
//                         "range", parameter("sampleRate").toReal()/2.0,
//                         "maxPeaks", parameter("maxPeaks"),
//                         "minPosition", parameter("minFrequency"),
//                         "maxPosition", parameter("maxFrequency"),
//                         "threshold", parameter("magnitudeThreshold"),
//                         "orderBy", orderBy);
_sineModelAnal->configure( "sampleRate", parameter("sampleRate").toReal(),
                            "maxnSines", parameter("maxnSines").toReal() ,
                            "freqDevOffset", parameter("freqDevOffset").toReal(),
                            "freqDevSlope",  parameter("freqDevSlope").toReal()
                            );

_sineModelSynth->configure( "sampleRate", parameter("sampleRate").toReal(),
                            "fftSize", parameter("frameSize").toReal(),
                            "hopSize", parameter("hopSize").toReal()
                            );

}



void SpsModelAnal::compute() {
  // inputs and outputs
  const std::vector<std::complex<Real> >& fft = _fft.get();

  std::vector<Real>& peakMagnitude = _magnitudes.get();
  std::vector<Real>& peakFrequency = _frequencies.get();
  std::vector<Real>& peakPhase = _phases.get();
  std::vector<Real>& stocEnv = _stocenv.get();

  std::vector<Real> fftmag;
  std::vector<Real> fftphase;

 _sineModelAnal->input("complex").set(fft);
 _sineModelAnal->output("magnitudes").set(peakMagnitude);
 _sineModelAnal->output("frequencies").set(peakFrequency);
 _sineModelAnal->output("phases").set(peakPhase);

  _sineModelAnal->compute();

  // compute stochastic envelope
  stochasticModelAnal(fft, peakMagnitude, peakFrequency, peakPhase, stocEnv);

}


// ---------------------------
// additional methods


void SpsModelAnal::stochasticModelAnal(const std::vector<std::complex<Real> > fftInput, const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<Real> &stocEnv)
{

  // subtract sines
  std::vector<std::complex<Real> > fftSines;
  std::vector<std::complex<Real> > fftRes;

  _sineModelSynth->input("magnitudes").set(magnitudes);
  _sineModelSynth->input("frequencies").set(frequencies);
  _sineModelSynth->input("phases").set(phases);
  _sineModelSynth->output("fft").set(fftSines);

  _sineModelSynth->compute();

  fftRes = fftInput; // initialize output

  for (int i= 0; i < (int)fftRes.size(); ++i)
  {
    fftRes[i].real(fftRes[i].real() - fftSines[i].real());
    fftRes[i].imag(fftRes[i].imag() - fftSines[i].imag());
  }

  // the decimation factor must be in a range (0.01 and 1) Default 0 0.2
  //Real stocf = ;
  Real stocf = std::min( std::max(0.01f, parameter("stocf").toReal()), 1.f);

  stocEnv.resize ( int( stocf * fftRes.size()));
  Real logMag = 0.;
  int decIdx = 0;
  //  std::fill(stocEnv.begin(), stocEnv.end(), 0.);
  for (int i=0; i< (int) stocEnv.size(); i++)
  {
    decIdx = int(0.5 + (i / stocf));
    logMag = log10( sqrt( fftRes[decIdx].real() * fftRes[decIdx].real() +  fftRes[decIdx].imag() * fftRes[decIdx].imag()));
    stocEnv[i] = std::max(-200., (20.f * std::log10(logMag + 1e-10)));
  }


}


