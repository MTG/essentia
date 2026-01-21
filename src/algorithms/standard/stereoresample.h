/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_STEREORESAMPLE_H
#define ESSENTIA_STEREORESAMPLE_H

//#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class StereoResample : public Algorithm {

 protected:
  Input<std::vector<StereoSample> > _signal;
  Output<std::vector<StereoSample> > _resampled;
  Algorithm* _stereoDemuxer;
  Algorithm* _stereoMuxer;
  Algorithm* _resample;

  double _factor;
  int _quality;
  std::vector<Real> _left, _right;
  std::vector<Real> _leftStorage, _rightStorage;

 public:
  StereoResample() {
    declareInput(_signal, "signal", "the input stereo signal");
    declareOutput(_resampled, "signal", "the resampled stereo signal");
  }

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality, 4 for fast linear approximation", "[0,4]", 1);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

/*
//#include "streamingalgorithm.h"
#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class StereoResample : public AlgorithmComposite {

 protected:
  Sink<StereoSample> _signal;
  Source<StereoSample> _resampled;

  Algorithm* _stereoDemuxer;
  Algorithm* _stereoMuxer;
  Algorithm* _resample;

  scheduler::Network* _network;

  bool _configured;
  int _preferredSize;

  void clearAlgos();

 public:
  StereoResample();
  ~StereoResample();

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input stereo signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output stereo signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality, 4 for fast linear approximation", "[0,4]", 1);
  }

  void configure();
  //AlgorithmStatus process();
  void reset();
  void createInnerNetwork();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia
*/


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class StereoResample : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<StereoSample> > _signal;
  Source<std::vector<StereoSample> > _resampled;

 public:

  StereoResample() {
    declareAlgorithm("StereoResample");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_resampled, TOKEN, "signal");

    // useless as we do it anyway in the configure() method
    //_StereoResampled.setBufferType(BufferUsage::forAudioStream);
  }

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STEREORESAMPLE_H
