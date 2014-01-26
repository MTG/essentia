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

#include "resample.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Resample::name = "Resample";
const char* Resample::description = DOC("This algorithm resamples the input signal to the desired sampling rate.\n\n"

"This algorithm is only supported if essentia has been compiled with Real=float, otherwise it will throw an exception. It may also throw an exception if there is an internal error in the SRC library during conversion.\n\n"

"References:\n"
"  [1] Secret Rabbit Code, http://www.mega-nerd.com/SRC\n\n"
"  [2] Resampling - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Resampling");


void Resample::configure() {
  _quality = parameter("quality").toInt();
  _factor = parameter("outputSampleRate").toReal() / parameter("inputSampleRate").toReal();

  // check to make sure Real is typedef'd as float
  if (sizeof(Real) != sizeof(float)) {
    throw EssentiaException("Resample: Error, Essentia has to be compiled with Real=float for resampling to work.");
  }
}

void Resample::compute() {
  const std::vector<Real>& signal = _signal.get();
  std::vector<Real>& resampled = _resampled.get();

  if (_factor == 1.0) {
    resampled = signal;
    return;
  }

  if (signal.empty()) return;

  SRC_DATA src;
  src.input_frames = (long)signal.size();
  src.data_in = const_cast<float*>(&(signal[0]));

  // add some samples to make sure we don't crash in a stupid way...
  src.output_frames = (long)((double)signal.size()*_factor + 100.0);
  resampled.resize(src.output_frames);
  src.data_out = &(resampled[0]);

  src.src_ratio = _factor;

  // do the conversion
  int error = src_simple(&src, _quality, 1);

  if (error) throw EssentiaException("Resample: Error in resampling: ", src_strerror(error));

  resampled.resize(src.output_frames_gen);
}

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {

const char* Resample::name = standard::Resample::name;
const char* Resample::description = standard::Resample::description;

// NOTE: streaming process differs slightly from the standard in that there is a transport delay inside the streaming version of the SRC converter: http://www.mega-nerd.com/SRC/faq.html#Q006. For this reason less amount of samples than the expected are found and thus the zeropadding at the end.

Resample::~Resample() {
  if (_state) src_delete(_state);
}

void Resample::configure() {
  int quality = parameter("quality").toInt();
  Real factor = parameter("outputSampleRate").toReal() / parameter("inputSampleRate").toReal();

  if (_state) src_delete(_state);
  int nChannels = 1;
  _state = src_new(quality, nChannels, &_errorCode);

  _data.src_ratio = factor;

  reset();
}

AlgorithmStatus Resample::process() {
  EXEC_DEBUG("process()");

  EXEC_DEBUG("Trying to acquire data");
  AlgorithmStatus status = acquireData();

  if (status != OK) {
    // FIXME: are we sure this still works?
    // if status == NO_OUTPUT, we should temporarily stop the resampler,
    // return from this function so its dependencies can process the frames,
    // and reschedule the framecutter to run when all this is done.
    if (status == NO_OUTPUT) {
      EXEC_DEBUG("no more output available for resampling; mark it for rescheduling and return");
      //_reschedule = true;
      return NO_OUTPUT; // if the buffer is full, we need to have produced something!
    }

    // if shouldStop is true, that means there is no more audio, so we need
    // to take what's left to fill in the output, instead of waiting for more
    // data to come in (which would have done by returning from this function)
    if (!shouldStop()) return NO_INPUT;

    int available = input("signal").available();
    EXEC_DEBUG("There are " << available << " available tokens");
    if (available == 0) return NO_INPUT;

    input("signal").setAcquireSize(available);
    input("signal").setReleaseSize(available);
    output("signal").setAcquireSize((int)(_data.src_ratio * available + 100 + (int)_delay));
    _data.end_of_input = 1;

    return process();
  }

  EXEC_DEBUG("data acquired");

  const vector<AudioSample>& signal = _signal.tokens();
  vector<AudioSample>& resampled = _resampled.tokens();

  EXEC_DEBUG("signal size:" << signal.size());
  EXEC_DEBUG("resampled size:" << resampled.size());

  _data.data_in = const_cast<float*>(&(signal[0]));
  _data.input_frames = (long)signal.size();

  _data.data_out = &(resampled[0]);
  _data.output_frames = (long)resampled.size();


  if (_data.src_ratio == 1.0) {
    assert(_data.output_frames >= _data.input_frames);
    fastcopy(_data.data_out, _data.data_in, _data.input_frames);
    _data.input_frames_used = _data.input_frames;
    _data.output_frames_gen = _data.input_frames;
  }
  else {
    int error = src_process(_state, &_data);

    if (error) {
      throw EssentiaException("Resample: ", src_strerror(error));
    }

    if (_data.input_frames_used == 0) {
      throw EssentiaException("Resample: Internal consumption problem while resampling");
    }
  }

  EXEC_DEBUG("input frames:" << _data.input_frames_used);
  EXEC_DEBUG("produced:" << _data.output_frames_gen);

  _delay += (Real)_data.input_frames_used*_data.src_ratio - (Real)_data.output_frames_gen;

  assert((int)resampled.size() >= _data.output_frames_gen);
  assert((int)signal.size() >= _data.input_frames_used);

  _signal.setReleaseSize(_data.input_frames_used);
  _resampled.setReleaseSize(_data.output_frames_gen);

  releaseData();

  EXEC_DEBUG("released");

  return OK;
}

void Resample::reset() {
  Algorithm::reset();
  _data.end_of_input = 0;
  _delay = 0;

  // make sure to reset I/O sizes, failure to do this causes inconsitent behavior with libsamplerate
  // my theory is that because the signal is being chopped up in different intervals than before,
  // the sampling is performed differently
  _signal.setAcquireSize(_preferredSize);
  _signal.setReleaseSize(_preferredSize);
  _resampled.setAcquireSize(_preferredSize);
  _resampled.setReleaseSize(_preferredSize);

  int maxElementsAtOnce = (int)(_data.src_ratio * _signal.acquireSize()) + 100;
  _resampled.setAcquireSize(maxElementsAtOnce);

  BufferInfo buf;
  buf.size = maxElementsAtOnce * 32;
  buf.maxContiguousElements = maxElementsAtOnce*2;
  _resampled.setBufferInfo(buf);

  int error = src_reset(_state);
  if (error) throw EssentiaException("Resample: ", src_strerror(error));
}

} // namespace streaming
} // namespace essentia
