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

#ifndef ESSENTIA_PITCHFILTERMAKAM_H
#define ESSENTIA_PITCHFILTERMAKAM_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchFilterMakam : public Algorithm {

 private:
  Input<std::vector<Real> > _energy;
  Input<std::vector<Real> > _pitch;
  Output<std::vector<Real> > _pitchFiltered;

  bool _octaveFilter;
  uint64_t _minChunkSize;

  bool areClose(Real num1, Real num2);
  void splitToChunks(const std::vector <Real>& pitch,
    std::vector <std::vector <Real> >& chunks,
    std::vector <uint64_t>& chunksIndexes,
    std::vector <uint64_t>& chunksSize);
  void joinChunks(const std::vector <std::vector <Real> >& chunks, std::vector <Real>& result);
  Real energyInChunk(const std::vector <Real>& energy, uint64_t chunkIndex, uint64_t chunkSize);
  void correctOctaveErrorsByChunks(std::vector <Real>& pitch);
  void removeExtremeValues(std::vector <Real>& pitch);
  void correctJumps(std::vector <Real>& pitch);
  void filterNoiseRegions(std::vector <Real>& pitch);
  void correctOctaveErrors(std::vector <Real>& pitch);
  void filterChunksByEnergy(std::vector <Real>& pitch, const std::vector <Real>& energy);

 public:
  PitchFilterMakam() {
    declareInput(_pitch, "pitch", "vector of pitch values for the input frames [Hz]");
    declareInput(_energy, "energy", "vector of energy values for the input frames");
    declareOutput(_pitchFiltered, "pitchFiltered", "vector of corrected pitch values [Hz]");
  }

  ~PitchFilterMakam() {
  };

  void declareParameters() {
    declareParameter("minChunkSize", "minumum number of frames in non-zero pitch chunks", "[0,inf)",  10);
    declareParameter("octaveFilter", "enable global octave filter", "{true,false}", false);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

}; // class PitchFilterMakam

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchFilterMakam : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _energy;
  Sink<std::vector<Real> > _pitch;
  Source<Real> _pitchFiltered;

 public:
  PitchFilterMakam() {
    declareAlgorithm("PitchFilterMakam");
    declareInput(_energy, TOKEN, "energy");
    declareInput(_pitch, TOKEN, "pitch");
    declareOutput(_pitchFiltered, TOKEN, "pitchFiltered");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHFILTERMAKAM_H
