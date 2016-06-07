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


#ifndef ESSENTIA_SYNTH_UTILS_H
#define ESSENTIA_SYNTH_UTILS_H

#include <essentia/algorithmfactory.h>


namespace essentia{


void scaleAudioVector(std::vector<Real> &buffer, const Real scale);
//void mixAudioVectors(const std::vector<Real> ina, const std::vector<Real> inb, const Real gaina, const Real gainb, std::vector<Real> &out);
void cleaningSineTracks(std::vector< std::vector<Real> >&freqsTotal, const int minFrames);
void genSpecSines(std::vector<Real> iploc, std::vector<Real> ipmag, std::vector<Real> ipphase, std::vector<std::complex<Real> > &outfft, const int fftSize);
void initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT);

} // namespace essentia

#endif // ESSENTIA_SYNTH_UTILS_H
