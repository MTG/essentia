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

#include "harmonicmask.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* HarmonicMask::name = "HarmonicMask";
const char* HarmonicMask::description = DOC("This algorithm applies a spectral mask to remove a pitched source component from the signal.\n"
                                        " It computes first an harmonic mask corresponding to the input pitch and applies the mask to the input FFT to remove that pitch. "
                                        "The bin width determines how many spectral bins are masked per harmonic partial.\n"
                                        "\n"
                                        "References:\n"
                                        " ");


void HarmonicMask::configure()
{

    _sampleRate = parameter("sampleRate").toInt();
    _binWidth = parameter("binWidth").toReal();


}

void HarmonicMask::compute()
{


    const std::vector<std::complex<Real> >& fft = _fft.get();
     const std::vector<Real> & pitchIn = _pitchIn.get();
    //const std::vector<Real >& pitch = _pitch.get(); // if input is a vector (Predominant)
    const Real& pitch = _pitch.get(); // input pitch is a scalar yinPitch

    std::vector<std::complex<Real> >& outfft = _outfft.get();

    int fftsize = fft.size();
    outfft.resize(fftsize);

    vector<Real> frequencies;
    vector<Real> magnitudes;

    // create mask
    vector<Real> mask;
    int maskSize = fftsize; // in Essentia the size of FFT output is (frameSize/2 +1).
    mask.resize(maskSize);
    int i, j;

    // init mask to ones
    for (i=0; i < fftsize; ++i)
    {
        mask[i] = 1.f;
    }

    // get pitch from input
    Real curPitchHz = 0;

    curPitchHz = pitch;

//cout << "cur pitch / size " << curPitchHz << ", " << pitchIn.size() << " ,";

    int nharmonic = 1;
    int cbin, lbin, rbin;

    while (curPitchHz > 0 && (nharmonic*curPitchHz < (_sampleRate / 2.f)))
    {

        cbin = floor(0.5 + (nharmonic * curPitchHz * 2 * fftsize )/ float(_sampleRate));
        lbin = cbin - _binWidth;
        rbin = cbin + _binWidth;
        lbin = max(0, lbin);
        rbin = min(rbin,maskSize -1);

        // set harmonic partials bins
        for (j=lbin; j<= rbin; ++j)
        {
            mask[j] = 0.f;
        }

        nharmonic++;
    }

    // apply TF
    for (int i=0; i < fftsize; ++i)
    {
        outfft[i] =  complex<Real> (fft[i].real() * mask[i], fft[i].imag()  * mask[i]); // real

    }


}
