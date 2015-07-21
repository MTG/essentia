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

#include "sinemodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SineModelAnal::name = "SineModelAnal";
const char* SineModelAnal::description = DOC("This algorithm computes the sine model analysis without sine tracking. It can process multiple FFT frames as input.");

// Initial implementation witohut tracking for both standard and steraming mode.
// Next step: implement sine tracking for standard implem entation, if we have access to all spectrogram.


void SineModel::processFrame(){

  // compute  sine freqs and phase for input frame
  		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		pmag = mX[ploc]                                       # get the magnitude of the peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
		ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz

}

void SineModel::sinusoidalTracking(){


}


void SineModel::compute() {

  const std::vector<std::vector<Real> >& spectrum = _spectrum.get();


  if (spectrum.size() < 2) {
    throw EssentiaException("SineModelAnal: input audio spectrum must be larger than 1 element");
  }





}
