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

#ifndef ESSENTIA_VAMPEASYWRAPPER_H
#define ESSENTIA_VAMPEASYWRAPPER_H

#include "vampwrapper.h"
#include <essentia/algorithmfactory.h>


#define WRAP_ALGO(algoname, unit, ndim, outputType)                      \
class algoname : public VampWrapper  {                                   \
public:                                                                  \
                                                                         \
  algoname(float sr) :                                                   \
    VampWrapper(essentia::standard::AlgorithmFactory::create(#algoname), sr) {} \
                                                                         \
  std::string getIdentifier() const  { return "essentia_" + info().name; }\
  std::string getName() const        { return info().name; }             \
                                                                         \
  OutputList getOutputDescriptors() const {                              \
    return genericDescriptor(unit, ndim);                                \
  }                                                                      \
                                                                         \
  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) { \
    if (getInputDomain() == FrequencyDomain) {                           \
      computeSpectrum(inputBuffers);                                     \
                                                                         \
      outputType value;                                                  \
                                                                         \
      _algo->input(_algo->inputNames()[0]).set(_spectrum);               \
      _algo->output(_algo->outputNames()[0]).set(value);                 \
                                                                         \
      _algo->compute();                                                  \
                                                                         \
      return returnFeature(value);                                       \
    }                                                                    \
    else {                                                               \
      std::cout << "ERROR: EasyWrapper not defined in temporal domain yet" << std::endl; \
      return FeatureSet();                                               \
    }                                                                    \
  }                                                                      \
}

#define WRAP_TEMPORAL_ALGO(algoname, unit, ndim, outputType)             \
class algoname : public VampWrapper  {                                   \
public:                                                                  \
                                                                         \
  algoname(float sr) :                                                   \
    VampWrapper(essentia::standard::AlgorithmFactory::create(#algoname), sr) {} \
                                                                         \
  std::string getIdentifier() const  { return "essentia_" + info().name; }\
  std::string getName() const        { return info().name; }             \
                                                                         \
                                                                         \
  OutputList getOutputDescriptors() const {                              \
    return genericDescriptor(unit, ndim);                                \
  }                                                                      \
                                                                         \
InputDomain getInputDomain() const { return TimeDomain; }                \
                                                                         \
  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) { \
    RogueVector<float> inputr(const_cast<float*>(inputBuffers[0]), _blockSize);\
    vector<float>& input = static_cast<vector<float>&>(inputr);\
                                                                         \
      outputType value;                                                  \
                                                                         \
      _algo->input(_algo->inputNames()[0]).set(input);               \
      _algo->output(_algo->outputNames()[0]).set(value);                 \
                                                                         \
      _algo->compute();                                                  \
                                                                         \
      return returnFeature(value);                                       \
  }                                                                      \
}

#define WRAP_PEAKS_ALGO(algoname, unit, ndim, outputType)                \
class algoname : public VampWrapper  {                                   \
public:                                                                  \
                                                                         \
  algoname(float sr) :                                                   \
    VampWrapper(essentia::standard::AlgorithmFactory::create(#algoname), sr) {} \
                                                                         \
  OutputList getOutputDescriptors() const {                              \
    return genericDescriptor(unit, ndim);                                \
  }                                                                      \
                                                                         \
  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) { \
    computePeaks(inputBuffers);                                          \
                                                                         \
    outputType value;                                                    \
                                                                         \
    _algo->input("magnitudes").set(_peakmags);                           \
    _algo->input("frequencies").set(_peakfreqs);                         \
    _algo->output(_algo->outputNames()[0]).set(value);                   \
                                                                         \
    _algo->compute();                                                    \
                                                                         \
    return returnFeature(value);                                         \
  }                                                                      \
}


#define WRAP_BARK_ALGO(algoname, unit, ndim, outputType)                           \
class B##algoname : public VampWrapper  {                                          \
public:                                                                            \
                                                                                   \
  B##algoname(float sr) :                                                          \
    VampWrapper(essentia::standard::AlgorithmFactory::create(#algoname), sr) {}    \
                                                                                   \
  std::string getIdentifier() const  { return std::string("bark_") + info().name; }\
  std::string getName() const        { return std::string("Bark ") + info().name; }\
                                                                         \
  OutputList getOutputDescriptors() const {                              \
    return genericDescriptor(unit, ndim, "bark_");                       \
  }                                                                      \
                                                                         \
  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) { \
    computeBarkBands(inputBuffers);                                      \
                                                                         \
    outputType value;                                                    \
                                                                         \
    _algo->input(_algo->inputNames()[0]).set(_barkBands);                \
    _algo->output(_algo->outputNames()[0]).set(value);                   \
                                                                         \
    _algo->compute();                                                    \
                                                                         \
    return returnFeature(value);                                         \
  }                                                                      \
}

#define WRAP_MEL_ALGO(algoname, unit, ndim, outputType)                            \
class M##algoname : public VampWrapper  {                                          \
public:                                                                            \
                                                                                   \
  M##algoname(float sr) :                                                          \
    VampWrapper(essentia::standard::AlgorithmFactory::create(#algoname), sr) {}    \
                                                                         \
  OutputList getOutputDescriptors() const {                              \
    return genericDescriptor(unit, ndim);                                \
  }                                                                      \
                                                                         \
  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) { \
    computeMelBands(inputBuffers);                                       \
                                                                         \
    outputType value;                                                    \
                                                                         \
    _algo->input(_algo->inputNames()[0]).set(_melBands);                 \
    _algo->output(_algo->outputNames()[0]).set(value);                   \
                                                                         \
    _algo->compute();                                                    \
                                                                         \
    return returnFeature(value);                                         \
  }                                                                      \
}
#endif // ESSENTIA_VAMPEASYWRAPPER_H
