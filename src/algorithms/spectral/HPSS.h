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

#ifndef ESSENTIA_HPSS_H
#define ESSENTIA_HPSS_H

#include "algorithm.h"
#include "essentiamath.h"
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>

namespace essentia {
namespace standard {

class HPSS : public Algorithm {
//  std::vector<Real> _harmonicMask, _percussiveMask; // TODO: I had to instantiate these variables inside the compute method
//  std::vector<Real> _enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum; // TODO: I had to instantiate these variables inside the compute method
 protected:
  Input<std::vector<Real>> _spectrum;
  Output<std::vector<Real>> _percussiveSpectrum;
  Output<std::vector<Real>> _harmonicSpectrum;


 public:
  HPSS() {
    declareInput(_spectrum, "spectrum", "spectrum frame");
    declareOutput(_percussiveSpectrum, "percussiveSpectrum", "spectrum of the percussive component");
    declareOutput(_harmonicSpectrum, "harmonicSpectrum", "spectrum of the harmonic component");
 
    // Algorithm Factory
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    _harmonicMedian  = factory.create("Median");
    _percussiveMedianFilter = factory.create("MedianFilter");
    output = factory.create("YamlOutput", "filename", "HPSS_debugging.yml");
  }

  ~HPSS() {
    output->input("pool").set(pool);
    output->compute();
    // if (_medianFilter) delete _medianFilter;
  }
  

  void declareParameters() {
    declareParameter("harmonicKernel", "kernel size of the harmonic median filter (positive odd number)",
    "(1,inf)", 17);
    declareParameter("kernelSize", "kernel size of the percussive median filter (positive odd number)",
    "(1, inf)", 17);
    declareParameter("harmonicMargin", "margin size for the harmonic mask", "[1,inf)", 1.f);
    declareParameter("percussiveMargin", "margin size for the percussive mask", "[1,inf)", 1.f);
    declareParameter("power", "the input frame size of the spectrum vector", "(0,inf)", 2.f);
    declareParameter("frameSize", "the input frame size of the spectrum vector", "(1,inf)", 1024);
  }

  void configure();
  void compute();
  void reset();

  void computingEnhancedSpectrograms(const std::vector<Real>& mX, std::vector<Real>& mP, std::vector<Real>& mH);
  void computingHarmonicEnhancedSpectrogram(const std::vector<Real>& mX, std::vector<Real>& mH);
  void computingPercussiveEnhancedSpectrogram(const std::vector<Real>& mX, std::vector<Real>& mP);
 

  void hardMasking(const std::vector<Real>& mX, const std::vector<Real>& mX_ref, std::vector<Real>& mask);
  void softMasking(const std::vector<Real>& mX, const std::vector<Real>& mX_ref, std::vector<Real>& mask);

  void computingResidual(const std::vector<Real>& mX, const std::vector<Real>& mP,const std::vector<Real>& mH,
                           std::vector<Real>& mR);


  static const char* name;
  static const char* category;
  static const char* description;
  static const Real precision;

 protected:  
  size_t _frameSize;
  int _harmonicKernel;
  Real _harmonicMargin, _percussiveMargin;
  Real _power;
  bool _split_zeros = false;

  // Harmonic Median 
  Algorithm *_harmonicMedian;
  std::vector<std::vector<Real>> _harmonicMatrixBuffer;
  Real _output_harmonicMedian;

  // Percussive Median Filter 
  Algorithm *_percussiveMedianFilter;

  Algorithm *output;

  Pool pool;

  void initialize();
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HPSS : public StreamingAlgorithmWrapper {

 protected:
  /*
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _harmonicSpectrum;
  Source<std::vector<Real> > _percussiveSpectrum;
  */

 public:
    HPSS() {
        declareAlgorithm("HPSS");
        /*
        declareInput(_spectrum, "spectrum", "spectrum frame");
        declareOutput(_percussiveSpectrum, "percussiveSpectrum", "spectrum of the percussive component");
        declareOutput(_harmonicSpectrum, "harmonicSpectrum", "spectrum of the harmonic component");
         */
    }

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HPSS_H
