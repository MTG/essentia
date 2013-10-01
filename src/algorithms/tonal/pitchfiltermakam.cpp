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

#include "pitchfiltermakam.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchFilterMakam::name = "PitchFilterMakam";
const char* PitchFilterMakam::version = "1.0";
const char* PitchFilterMakam::description = DOC("This algorithm corrects the fundamental frequency estimations for a sequence of frames. The original estimations can be computed with the PitchYinFFT algorithm.\n"
"\n"
"The algorithm is based on the code of Makam Toolbox 1.0.\n"
"ftp://ftp.iyte.edu.tr/share/ktm-nota/TuningMeasurement.html\n"
"\n"
"References:\n"
"  [1] B. Bozkurt, \"An Automatic Pitch Analysis Method for Turkish Maqam\n"
"  Music,\" Journal of New Music Research. 37(1), 1-13.\n");

void PitchFilterMakam::configure() {
  _minChunkSize = parameter("minChunkSize").toInt();
}

void PitchFilterMakam::compute() {
  const vector<Real>& pitch = _pitch.get();
  const vector<Real>& energy = _energy.get();

  // sanity checks, pitch and energy values should be non-negative
  if (pitch.size() != energy.size())
      throw EssentiaException("PitchFilterMakam: Pitch and energy vectors should be of the same size.");
  if (pitch.size() == 0)
      throw EssentiaException("PitchFilterMakam: Pitch and energy vectors are empty.");
  for (size_t i=0; i<pitch.size(); i++) {
    if (pitch[i] < 0)
      throw EssentiaException("PitchFilterMakam: Pitch values should be non-negative.");
    if (energy[i] < 0)
      throw EssentiaException("PitchFilterMakam: Energy values should be non-negative.");
  }

  vector <Real>& pitchFiltered = _pitchFiltered.get();

  pitchFiltered = pitch;
  correctOctaveErrorsByChunks(pitchFiltered);
  removeExtremeValues(pitchFiltered);

  // correct pitch curve jumps in both directions (forwards, backwards)
  correctJumps(pitchFiltered);
  reverse(pitchFiltered.begin(), pitchFiltered.end());
  correctJumps(pitchFiltered);
  reverse(pitchFiltered.begin(), pitchFiltered.end());

  filterNoiseRegions(pitchFiltered);

  if (_octaveFilter) {
    // correct octave errors of pitch curve in both direction (forwards, backwards)
    correctOctaveErrors(pitchFiltered);
    reverse(pitchFiltered.begin(), pitchFiltered.end());
    correctOctaveErrors(pitchFiltered);
    reverse(pitchFiltered.begin(), pitchFiltered.end());
  }
  correctOctaveErrorsByChunks(pitchFiltered);

  filterChunksByEnergy(pitchFiltered, energy);
}

bool PitchFilterMakam::areClose(Real num1, Real num2) {
  Real d = fabs(num1 - num2);
  Real av = (num1 + num2) / 2;

  if (av == 0)  // num1 and num2 >= 0, thus num1 = num2 = 0
    return true;
  else if (d/av < 0.2)
    return true;
  else
    return false;
}

void PitchFilterMakam::splitToChunks(const vector <Real>& pitch,
    vector <vector <Real> >& chunks,
    vector <uint64_t>& chunksIndexes,
    vector <uint64_t>& chunksSize) {

    // populate chunks
    for (size_t i=0; i<pitch.size(); i++) {
        Real pitch_interval;

        if (i==0)
            pitch_interval = -1;
        else if (i==pitch.size()-1) // add last element to the last chunk
            pitch_interval = 1;
        else
            pitch_interval = pitch[i] / pitch[i-1];

        if (pitch_interval < 0.80 || pitch_interval > 1.2) {
            // add to a new chunk
            vector <Real> new_chunk;
            new_chunk.push_back(pitch[i]);
            chunks.push_back(new_chunk);
            chunksIndexes.push_back(i);
        } else {
            // add to old chunk
            chunks.back().push_back(pitch[i]);
        }
    }
    // compute chunk lengths
    for (size_t i=0; i < chunks.size(); i++) {
        chunksSize.push_back(chunks[i].size());
    }
}

void PitchFilterMakam::joinChunks(const vector <vector <Real> >& chunks, vector <Real>& result) {
  result.clear();
  for (size_t i=0; i<chunks.size(); i++) {
    result.insert(result.end(), chunks[i].begin(), chunks[i].end());
  }
}

Real PitchFilterMakam::energyInChunk(const vector <Real>& energy, uint64_t chunkIndex, uint64_t chunkSize) {
  return accumulate(energy.begin() + chunkIndex, energy.begin() + chunkIndex + chunkSize, 0.0) / chunkSize;
}

void PitchFilterMakam::correctOctaveErrorsByChunks(vector <Real>& pitch) {
  vector <vector <Real> > chunks;
  vector <uint64_t> chunksIndexes;
  vector <uint64_t> chunksSize;

  // split pitch values vector to chunks
  splitToChunks(pitch, chunks, chunksIndexes, chunksSize);
  // correct octave errors in chunks
  for (size_t i=1; i < chunks.size()-1; i++) {
    if (chunks[i].size() < chunks[i-1].size() || chunks[i].size() < chunks[i+1].size()) {
      Real octaveTranspose = 1.;

      // check if transpose is needed
      if (areClose(chunks[i].front() / 2, chunks[i-1].back()) && chunks[i].back() / 1.5 > chunks[i+1].front())
        octaveTranspose = 0.5; // 1 octave down
      else if (areClose(chunks[i].back() / 2, chunks[i+1].front()) && chunks[i].front() / 1.5 > chunks[i-1].back())
        octaveTranspose = 0.5;
      else if (areClose(chunks[i].front() * 2, chunks[i-1].back()) && chunks[i].back() * 1.5 < chunks[i+1].front())
        octaveTranspose = 2.;  // 1 octave up
      else if (chunks[i].front() * 1.5 < chunks[i-1].back() && areClose(chunks[i].back() * 2, chunks[i+1].front()))
        octaveTranspose = 2.;

      // transpose
      if (octaveTranspose != 1.) {
        for (size_t k=0; k<chunks[i].size(); k++) {
          chunks[i][k] *= octaveTranspose;
        }
      }
    }
  }
  // merge chunks
  joinChunks(chunks, pitch);
}

void PitchFilterMakam::removeExtremeValues(vector <Real>& pitch) {
  // compute pitch statistics
  Real pitchMax = pitch[argmax(pitch)];
  Real pitchMean = mean(pitch);
  Real pitchStddev = stddev(pitch, pitchMean);

  // compute pitch histogram [0; maximum pitch] with 99 bins
  const int nbins = 99;
  vector <int> binValues(nbins);
  vector <Real> binCenters(nbins);

  // FIXME is int range enough for extremly long audio input?
  hist(&pitch[0], pitch.size(), &binValues[0], &binCenters[0], nbins);

  // find the first zero bin which is followed by another zero bin
  // if it exists, check if a cumulative sum of the bins before it is larger than 90% of total cumulative sum
  // if so, the rest of the bins are high-frequency outliers
  // set the true maximum pitch to the upper edge of this zero bin

  for(size_t i=0; i<binValues.size()-1; i++) {
    if (binValues[i] == 0 && binValues[i+1] == 0) {
      if (accumulate(binValues.begin(), binValues.begin() + i, 0) > 0.9 * accumulate(binValues.begin(), binValues.end(), 0)) {
        pitchMax = binCenters[i]; // FIXME original algorithm uses upper edge of the bin
        break;
      }
    }
  }

  Real pitchMaxCandidate = fmax(pitchMean*4, pitchMean + 2*pitchStddev);
  pitchMax = fmin(pitchMax, pitchMaxCandidate);

  // zero pitch values greater than the estimated maximum
  for(size_t i=0; i<pitch.size(); i++)
    if (pitch[i] > pitchMax)
      pitch[i] = 0;

  pitchMean = mean(pitch); // recompute mean again (following the original algorithm)
  Real pitchMin = pitchMean / 4;

  // zero pitch values less than the estimated minimum
  for(size_t i=0; i<pitch.size(); i++)
    if (pitch[i] < pitchMin)
      pitch[i] = 0;
}

void PitchFilterMakam::correctJumps(vector <Real>& pitch) {
  // corrects jumps/discontinuities within the pitch curve
  for (size_t i=4; i<pitch.size()-6; i++) {
    // if four previous values form continuous curve
    if (areClose(pitch[i-4], pitch[i-3]) && areClose(pitch[i-3], pitch[i-2]) && areClose(pitch[i-2], pitch[i-1])) {

      // quadriple point
      if (areClose(pitch[i+4], pitch[i+5]) && areClose(pitch[i+5], pitch[i+6])) {
        if (!areClose(pitch[i], pitch[i-1]) && !areClose(pitch[i], pitch[i+4]))
          pitch[i] = pitch[i-1];
        if (!areClose(pitch[i+3], pitch[i-1]) && !areClose(pitch[i+3], pitch[i+4]))
          pitch[i+3]=pitch[i+4];
      }

      // triple point
      if (areClose(pitch[i+3], pitch[i+4]) && areClose(pitch[i+4], pitch[i+5])) {
        if (!areClose(pitch[i], pitch[i-1]) && !areClose(pitch[i], pitch[i+3]))
          pitch[i] = pitch[i-1];
        if (!areClose(pitch[i+2], pitch[i-1]) && !areClose(pitch[i+2], pitch[i+3]))
          pitch[i+2] = pitch[i+3];
      }

      // double point
      if (areClose(pitch[i+2], pitch[i+3]) && areClose(pitch[i+3], pitch[i+4])) {
        if (!areClose(pitch[i], pitch[i-1]) && !areClose(pitch[i], pitch[i+2]))
          pitch[i] = pitch[i-1];
        if (!areClose(pitch[i+1], pitch[i-1]) && !areClose(pitch[i+1], pitch[i+2]))
          pitch[i+1] = pitch[i+2];
      }

      // single point
      if (areClose(pitch[i+1], pitch[i+2]) && areClose(pitch[i+2], pitch[i+3]))
        if (!areClose(pitch[i], pitch[i-1]) && !areClose(pitch[i], pitch[i+1]))
          pitch[i] = pitch[i-1];
    }
  }
}

void PitchFilterMakam::filterNoiseRegions(vector <Real>& pitch) {
  // assign zero frequency to noisy pitch regions in three rounds
  // in original algorithm, frequency of 8.17579891564371 Hz, refered as 'zero cent frequency',  is used
  for (int m=0; m<3; m++) {

    // 3 separate points, assign 0 to mid
    for (size_t i=1; i<pitch.size()-2; i++) {
      if (!areClose(pitch[i-1], pitch[i]) && !areClose(pitch[i], pitch[i+1]))
        pitch[i] = 0;
    }

    // 4 points, 2 in mid are similar but different from the boundaries, assign 0 to mid two
    for (size_t i=2; i<pitch.size()-3; i++) {
      if (!areClose(pitch[i-2], pitch[i]) && !areClose(pitch[i-1], pitch[i])
          && !areClose(pitch[i+1], pitch[i+2]) && !areClose(pitch[i+1], pitch[i+3])) {
        pitch[i] = 0;
        pitch[i+1] = 0;
      }
    }
  }

  // filter out noise like variations
  for (size_t i=1; i<pitch.size()-2; i++) {
    if (!areClose(pitch[i-1], pitch[i])
        && !areClose(pitch[i], pitch[i+1])
        && !areClose(pitch[i+1], pitch[i+2])
        && !areClose(pitch[i-1], pitch[i+1])
        && !areClose(pitch[i], pitch[i+2])
        && !areClose(pitch[i-1], pitch[i+2])) {
          pitch[i] = 0;
          pitch[i+1] = 0;
        }
  }
}

void PitchFilterMakam::correctOctaveErrors(vector <Real>& pitch) {
  Real pitchMid = (median(pitch)+ mean(pitch)) / 2;
  for (size_t i=4; i<pitch.size()-2; i++) {
    // if previous values are continuous
    if (areClose(pitch[i-1], pitch[i-2]) && areClose(pitch[i-2], pitch[i-3]) && areClose(pitch[i-3], pitch[i-4])) {
      if (pitch[i] > pitchMid * 1.8) {
        if (areClose(pitch[i-1], pitch[i]/2))
          pitch[i] /= 2;
        else if (areClose(pitch[i-1], pitch[i]/4))
          pitch[i] /= 4;
      } else if (pitch[i] < pitchMid / 1.8) {
        if (areClose(pitch[i-1], pitch[i]*2))
          pitch[i] *= 2;
        else if (areClose(pitch[i-1], pitch[i]*4))
          pitch[i] *= 4;
      }
    }
  }
}

void PitchFilterMakam::filterChunksByEnergy(std::vector <Real>& pitch, const std::vector <Real>& energy) {
  // original algorithm uses average signal amplitude instead of energy
  // short chunks with average amplitude, less than 1/6 of average energy of the longest chunk, are filtered
  // we, instead, use spectral energy

  vector <vector <Real> > chunks;
  vector <uint64_t> chunksIndexes;
  vector <uint64_t> chunksSize;

  // split pitch values vector to chunks
  splitToChunks(pitch, chunks, chunksIndexes, chunksSize);

  // compute average energy of the chunk with maximum length
  size_t max_i = max_element(chunksSize.begin(), chunksSize.end()) - chunksSize.begin();
  Real energyInLongestChunk = energyInChunk(energy, chunksIndexes[max_i], chunksSize[max_i]);
  Real energyMinLimit = energyInLongestChunk / 36; // corresponds to squared average amplitude  FIXME make it a parameter

  for (size_t i=0; i<chunks.size(); i++) {
    // check only non-zero pitch chunks
    if (chunks[i][argmax(chunks[i])] > 0) {
      // cout << "chunk with non-zero pitch, " << "size=" << chunksSize[i] << ", energy=" << energyInChunk(energy, chunksIndexes[i], chunksSize[i]) << endl;
      if ((chunksSize[i] < _minChunkSize) ||
          (chunksSize[i] < _minChunkSize*3 && energyInChunk(energy, chunksIndexes[i], chunksSize[i]) < energyMinLimit)) {
        for (size_t k=0; k<chunks[i].size(); k++)
          chunks[i][k] = 0.0;
      }
    }
  }

  // recompose chunks
  joinChunks(chunks, pitch);
}

