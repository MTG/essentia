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

#ifndef ESSENTIA_PITCHCONTOURSEGMENTATION_H
#define ESSENTIA_PITCHCONTOURSEGMENTATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class PitchContourSegmentation : public Algorithm {

 private:
  Input<std::vector<Real> > _pitch;
  Input<std::vector<Real> > _signal;
  Output<Real> _onset;
  Output<Real> _duration;
  Output<int> _MIDIpitch;

 public:
  PitchContourSegmentation() {
    declareInput(_pitch, "pitch", "estimated pitch contour [Hz]");
    declareInput(_signal, "signal", "input audio signal");
    declareOutput(_onset, "onset", "note onset times [s]");
    declareOutput(_duration, "duration", "note durations [s]");
    declareOutput(_MIDIpitch, "MIDIpitch", "quantized MIDI pitch value");
  }

  void declareParameters() {
    declareParameter("minDur", "minimum note duration [s]", "(0,inf)", 0.05);
    declareParameter("tuningFreq", "tuning reference frequency  [Hz]", "(0,22000)", 440);
  }

  void compute();
  void configure();
  void reset();

  static const char* name;
  static const char* description;

 protected:
  Real _minDur;
  Real _tuningFreq;
    /*
  std::vector<Real> _histogram;
  std::vector<Real> _globalHistogram;
*/
    /*
  Real currentTuningCents() const;
  Real tuningFrequencyFromCents(Real cents) const;
  void updateOutputs();
     */
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchContourSegmentation : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _pitch;
  Sink<std::vector<Real> > _signal;
  Source<Real> _onset;
  Source<Real> _duration;
  Source<Real> _MIDIpitch;

 public:
  PitchContourSegmentation() {
    declareAlgorithm("PitchContourSegmentation");
    declareInput(_pitch, TOKEN, "pitch");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_onset, TOKEN, "onset");
    declareOutput(_duration, TOKEN, "duration");
    declareOutput(_MIDIpitch, TOKEN, "MIDIpitch");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TUNINGFREQUENCY_H
