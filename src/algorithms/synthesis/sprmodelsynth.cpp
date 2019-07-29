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

#include "sprmodelsynth.h"
#include "essentiamath.h"


using namespace essentia;
using namespace standard;


const char* SprModelSynth::name = "SprModelSynth";
const char* SprModelSynth::category = "Synthesis";
const char* SprModelSynth::description = DOC("This algorithm computes the sinusoidal plus residual model synthesis from SPS model analysis.");



// configure
void SprModelSynth::configure()
{
  _sampleRate = parameter("sampleRate").toReal();
  _fftSize = parameter("fftSize").toInt();
  _hopSize = parameter("hopSize").toInt();

  _sineModelSynth->configure("sampleRate", _sampleRate,
                            "fftSize", _fftSize,
                            "hopSize", _hopSize);


  _ifftSine->configure("size", _fftSize);

  _overlapAdd->configure("frameSize", _fftSize, // uses synthesis window
                         "hopSize", _hopSize);

}


void SprModelSynth::compute() {

  const std::vector<Real>& magnitudes = _magnitudes.get();
  const std::vector<Real>& frequencies = _frequencies.get();
  const std::vector<Real>& phases = _phases.get();
  const std::vector<Real>& res = _res.get();

  std::vector<Real>& outframe = _outframe.get();
  std::vector<Real>& outsineframe = _outsineframe.get();
  std::vector<Real>& outresframe = _outresframe.get();


  // temp vectors
  std::vector<std::complex<Real> > fftSines;
  std::vector<Real> wsineFrame; // windowed frames
  std::vector<Real> sineFrame;  // overlap output frame
  std::vector<Real> resFrame;
  
    int i = 0;
  // copy input residual to output residual frame
  for (i=0; i < (int)  res.size();  i++)
    resFrame.push_back(res[i]);  



  _sineModelSynth->input("magnitudes").set(magnitudes);
  _sineModelSynth->input("frequencies").set(frequencies);
  _sineModelSynth->input("phases").set(phases);
  _sineModelSynth->output("fft").set(fftSines);

  _sineModelSynth->compute();

  _ifftSine->input("fft").set(fftSines);
  _ifftSine->output("frame").set(wsineFrame);
  _ifftSine->compute();
  _overlapAdd->input("signal").set(wsineFrame);
  _overlapAdd->output("signal").set(sineFrame);
  _overlapAdd->compute();

  // add sine and residual  copmponents
 outframe.clear();
 outsineframe.clear();
 outresframe.clear();
 for (i = 0; i < _hopSize; ++i)
  {
    outframe.push_back(sineFrame[i] + resFrame[i]);
    outsineframe.push_back(sineFrame[i]) ;
    outresframe.push_back(resFrame[i]);
  }

}

