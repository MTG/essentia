#ifndef ESSENTIA_VAMPEASYWRAPPER_H
#define ESSENTIA_VAMPEASYWRAPPER_H

#include "vampwrapper.h" 
#include "algorithmfactory.h"


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
