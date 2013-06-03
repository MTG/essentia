#include "vampwrapper.h"
#include <cmath>
#include <iostream>
using namespace std;
using namespace essentia;

int VampWrapper::essentiaVampPluginId = 0;

VampWrapper::VampWrapper(standard::Algorithm* algo, float inputSampleRate)
  : Vamp::Plugin(inputSampleRate), _algo(algo), _sampleRate(inputSampleRate) {

  try {
    _algo->configure("sampleRate", inputSampleRate);
  }
  catch (const EssentiaException&) {
    ;
  }
  _pluginId = essentiaVampPluginId++;
  setName(info().name);
  setDescription(info().description);
}


VampWrapper::~VampWrapper() { delete _algo; }

bool VampWrapper::initialise(size_t channels, size_t stepSize, size_t blockSize) {
  if (channels > 1) return false;

  _stepSize = (int)stepSize;
  _blockSize = (int)blockSize;

  _spectrum.resize(_blockSize/2+1);
  _phase.resize(_blockSize/2+1);

  _peaks = standard::AlgorithmFactory::create("SpectralPeaks",
                                              "sampleRate", _sampleRate,
                                              "orderBy", "frequency");
  _peaks->input("spectrum").set(_spectrum);
  _peaks->output("magnitudes").set(_peakmags);
  _peaks->output("frequencies").set(_peakfreqs);

  _bbands = standard::AlgorithmFactory::create("BarkBands",
                                               "numberBands", 20);
  _bbands->input("spectrum").set(_spectrum);
  _bbands->output("bands").set(_barkBands);

  _mbands = standard::AlgorithmFactory::create("MelBands",
                                               "numberBands", 24);
  _mbands->input("spectrum").set(_spectrum);
  _mbands->output("bands").set(_melBands);

  try { _algo->configure("frameSize", _blockSize); } catch (...) {}
  try { _algo->configure("hopSize", _stepSize); } catch (...) {}

  return true;
}

void VampWrapper::reset() { _algo->reset(); }


void VampWrapper::computeSpectrum(const float *const *inputBuffers) {
  const float* fft = inputBuffers[0];
  for (int i=0; i<=_blockSize/2; i++) {
    _spectrum[i] = sqrt(fft[2*i] * fft[2*i] + fft[2*i+1] * fft[2*i+1]);
    _phase[i] = atan2(fft[2*i+1],fft[2*i]);
  }
}

void VampWrapper::computePeaks(const float *const *inputBuffers) {
  computeSpectrum(inputBuffers);
  _peaks->compute();
}

void VampWrapper::computeBarkBands(const float *const *inputBuffers) {
  computeSpectrum(inputBuffers);
  _bbands->compute();
}

void VampWrapper::computeMelBands(const float *const *inputBuffers) {
  computeSpectrum(inputBuffers);
  _mbands->compute();
}

VampWrapper::OutputList
VampWrapper::genericDescriptor(const std::string& unit,
                               int ndim, const string& prefix) const {
  OutputList list;

  OutputDescriptor d;
  d.identifier = prefix + getIdentifier();
  d.name = prefix + getName();
  d.description = prefix + getDescription();
  d.unit = unit;
  d.hasFixedBinCount = true;
  d.binCount = ndim;
  d.hasKnownExtents = false;
  d.isQuantized = false;
  d.sampleType = OutputDescriptor::OneSamplePerStep;
  list.push_back(d);

  return list;
}


VampWrapper::Feature VampWrapper::makeFeature(float f) const {
  Feature feat;
  feat.hasTimestamp = false;
  feat.values.push_back(f);
  return feat;
}

VampWrapper::Feature VampWrapper::makeFeature(const vector<float>& f) const {
  Feature feat;
  feat.hasTimestamp = false;
  feat.values = f;
  return feat;
}
