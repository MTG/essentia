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

#include <deque>
#include <essentia/essentiamath.h>
#include "vamp/vamp.h"
#include "vamp-sdk/PluginAdapter.h"
#include "vampeasywrapper.h"
#include <essentia/utils/tnt/tnt.h>
#include <essentia/utils/tnt/tnt2vector.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/streaming/algorithms/vectoroutput.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>

using namespace std;
using namespace essentia;
using namespace essentia::scheduler;

class Pitch : public VampWrapper  {
public:

  Pitch(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("PitchYinFFT"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "pitch";
    d.name = "Pitch";
    d.description = "pitch";
    d.unit = "Hz";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "pitchconfidence";
    d.name = "Pitch confidence";
    d.description = "gloub";
    d.unit = "";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;
    ParameterDescriptor d;

    d.identifier = "minFrequency";
    d.name = "minFrequency";
    d.description = "the minimum allowed frequency";
    d.unit = "Hz";
    d.minValue = 1;
    d.maxValue = _sampleRate/2.0-1;
    d.defaultValue = 20;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "maxFrequency";
    d.name = "maxFrequency";
    d.description = "the maximum allowed frequency";
    d.unit = "Hz";
    d.minValue = 1;
    d.maxValue = _sampleRate/2.0;
    d.defaultValue = _sampleRate/2.0;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "interpolate";
    d.name = "interpolate";
    d.description = "Use parabolic interpolation to find peaks";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 1;
    d.defaultValue = 1;
    d.isQuantized = true;
    d.valueNames.push_back("False");
    d.valueNames.push_back("True");
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    if (id == "interpolate") {
      if (_algo->parameter("interpolate").toBool()) return 1;
      else return 0;
    }
    else return _algo->parameter(id).toReal();
  }

  void setParameter(string id, float value) {
    const vector<ParameterDescriptor>& params = getParameterDescriptors();
    ParameterMap parameterMap;
    float interpolate_value;
    for (int i=0; i < (int)params.size(); i++) {
      if (params[i].identifier == "interpolate") {
        if (params[i].identifier == id) interpolate_value = value;
        else interpolate_value = getParameter(params[i].identifier);
        if (interpolate_value == 0.0) parameterMap.add("interpolate", false);
        else parameterMap.add("interpolate", true);
      }
      else if (params[i].identifier == id) parameterMap.add(id, value);
      else parameterMap.add(params[i].identifier, getParameter(params[i].identifier));
      }
    _algo->configure(parameterMap);
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    float pitch, pitchconf;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("pitch").set(pitch);
    _algo->output("pitchConfidence").set(pitchconf);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(pitch));
    result[1].push_back(makeFeature(pitchconf));

    return result;
  }
};


class DistributionShape : public VampWrapper  {
  standard::Algorithm* _cmoments;
  vector<float> _moments;
public:

  DistributionShape(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("DistributionShape"), sr) {
    _cmoments = standard::AlgorithmFactory::create("CentralMoments");
    _cmoments->input("array").set(_spectrum);
    _cmoments->output("centralMoments").set(_moments);
  }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "spread";
    d.name = "spread";
    d.description = info().description;
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = d.name = "skewness";
    list.push_back(d);

    d.identifier = d.name = "kurtosis";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    float spread, skewness, kurtosis;

    _cmoments->compute();

    _algo->input("centralMoments").set(_moments);
    _algo->output("spread").set(spread);
    _algo->output("skewness").set(skewness);
    _algo->output("kurtosis").set(kurtosis);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(spread));
    result[1].push_back(makeFeature(skewness));
    result[2].push_back(makeFeature(kurtosis));

    return result;
  }
};


class BarkShape : public VampWrapper  {
  standard::Algorithm* _cmoments;
  vector<float> _moments;
public:

  BarkShape(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("DistributionShape"), sr) {
    _cmoments = standard::AlgorithmFactory::create("CentralMoments");
    _cmoments->input("array").set(_barkBands);
    _cmoments->output("centralMoments").set(_moments);
  }

  string getIdentifier() const { return "barkshape"; }
  string getName() const { return "BarkShape"; }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "spread";
    d.name = "spread";
    d.description = info().description;
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = d.name = "skewness";
    list.push_back(d);

    d.identifier = d.name = "kurtosis";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeBarkBands(inputBuffers);

    float spread, skewness, kurtosis;

    _cmoments->compute();

    _algo->input("centralMoments").set(_moments);
    _algo->output("spread").set(spread);
    _algo->output("skewness").set(skewness);
    _algo->output("kurtosis").set(kurtosis);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(spread));
    result[1].push_back(makeFeature(skewness));
    result[2].push_back(makeFeature(kurtosis));

    return result;
  }
};


class SpectralContrast : public VampWrapper  {
public:

  SpectralContrast(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("SpectralContrast"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "spectralcontrast";
    d.name = "Spectral contrast";
    d.description = "Spectral contrast";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = _algo->parameter("numberBands").toInt();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "spectralvalleys";
    d.name = "Spectral valleys";
    d.description = "Spectral valleys";
    d.unit = "";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    vector<float> contrast, valleys;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("spectralContrast").set(contrast);
    _algo->output("spectralValley").set(valleys);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(contrast));
    result[1].push_back(makeFeature(valleys));

    return result;
  }
};

class HFC: public VampWrapper  {
public:

  HFC(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("HFC", "type", "Brossier"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "hfc";
    d.name = "HFC";
    d.description = "high frequency content";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "type";
    d.name = "type";
    d.description = "the type of HFC";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 2;
    d.defaultValue = 1;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("Brossier");
    d.valueNames.push_back("Masri");
    d.valueNames.push_back("Jensen");
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    if (_algo->parameter("type") == "Brossier") return 0;
    if (_algo->parameter("type") == "Masri") return 1;
    return 2;
  }

  void setParameter(string id, float value) {
    if (int(value) == 0) _algo->configure("type", "Brossier");
    else if (int(value) == 1) _algo->configure("type", "Masri");
    else _algo->configure("type", "Jensen");
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    Real hfc;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("hfc").set(hfc);

    _algo->compute();

    return returnFeature(hfc);
  }
};

class OnsetDetection : public VampWrapper  {
public:

  OnsetDetection(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("OnsetDetection"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "onsetdetection";
    d.name = "onset detection";
    d.description = "onset detection function";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "method";
    d.name = "method";
    d.description = "the method used for onset detection";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 1;
    d.defaultValue = 1;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("hfc");
    d.valueNames.push_back("complex");
    d.valueNames.push_back("complex_phase");
    d.valueNames.push_back("flux");
    d.valueNames.push_back("melflux");
    d.valueNames.push_back("rms");
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    if (_algo->parameter("method")=="hfc") return 0;
    else if (_algo->parameter("method")=="complex") return 1;
    else if (_algo->parameter("method")=="complex_phase") return 2;
    else if (_algo->parameter("method")=="flux") return 3;
    else if (_algo->parameter("method")=="melflux") return 4;
    else if (_algo->parameter("method")=="rms") return 5;
    else {
        E_WARNING("OnsetDetection vamp wrapper: unknown parameter '" << id << "'");
        return -1;
    }
  }

  void setParameter(string id, float value) {
    switch(int(value)) {
      case 0:
        _algo->configure("method", "hfc");
        break;
      case 1:
        _algo->configure("method", "complex");
        break;
      case 2:
        _algo->configure("method", "complex_phase");
        break;
      case 3:
        _algo->configure("method", "flux");
        break;
      case 4:
        _algo->configure("method", "melflux");
        break;
      case 5:
        _algo->configure("method", "rms");
        break;
    }
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    Real onsetDetection;

    _algo->input("spectrum").set(_spectrum);
    _algo->input("phase").set(_phase);
    _algo->output("onsetDetection").set(onsetDetection);

    _algo->compute();

    return returnFeature(onsetDetection);
  }
};

class Onsets: public VampWrapper  {
public:

  Onsets(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("OnsetDetection", "method", "complex"), sr) {
      // use the following functions to ovewrite the values given by essentia,
      // so it doesn't collide with OnsetDetection name & description.
      AlgorithmInfo<standard::Algorithm> info = standard::AlgorithmFactory::getInfo("Onsets");
      setName(info.name);
      setDescription(info.description);
    }


  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "onsets";
    d.name = "onsets";
    d.description = "note onset locations";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 0;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleRate = _sampleRate;
    d.sampleType = OutputDescriptor::VariableSampleRate;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime ts) {
    standard::Algorithm * hfc = standard::AlgorithmFactory::create("OnsetDetection", "method", "hfc");
    Real complexDetection, hfcDetection;

    computeSpectrum(inputBuffers);

    _algo->input("spectrum").set(_spectrum);
    _algo->input("phase").set(_phase);
    _algo->output("onsetDetection").set(complexDetection);
    hfc->input("spectrum").set(_spectrum);
    hfc->input("phase").set(_phase);
    hfc->output("onsetDetection").set(hfcDetection);

    _algo->compute();
    hfc->compute();

    _pool.add("complex_detection", complexDetection);
    _pool.add("hfc_detection", hfcDetection);
    delete hfc;
    //float timeStamp = ts - Vamp::RealTime::frame2RealTime(_stepSize, lrintf(_sampleRate));
    //_pool.add("time_stamp", timeStamp);

    return FeatureSet();
  }

  FeatureSet getRemainingFeatures() {
    // Time onsets
    TNT::Array2D<Real> detections;
    vector<Real> hfc = _pool.value<vector<Real> >("hfc_detection");
    vector<Real> complexdomain = _pool.value<vector<Real> >("complex_detection");
    detections = TNT::Array2D<Real>(2, hfc.size());

    for (int i=0; i<int(hfc.size()); ++i) {
      detections[0][i] = hfc[i];
      detections[1][i] = complexdomain[i];
    }
    vector<Real> weights(2);
    weights[0] = 1.0;
    weights[1] = 1.0;

    vector<Real> onsetTimes;

    standard::Algorithm * onsets = standard::AlgorithmFactory::create("Onsets");
    onsets->input("detections").set(detections);
    onsets->input("weights").set(weights);
    onsets->output("onsets").set(onsetTimes);
    onsets->compute();
    _pool.remove("hfc_detection");
    _pool.remove("complex_detection");
    delete onsets;

    FeatureSet result;
    vector<Real>::const_iterator it = onsetTimes.begin();
    while (it != onsetTimes.end()) {
      Feature onset;
      onset.hasTimestamp = true;
      onset.timestamp = Vamp::RealTime::fromSeconds(*it);
      result[0].push_back(onset);
      ++it;
    }

    return result;
  }

};

class RhythmTransform : public VampWrapper  {
public:

  RhythmTransform(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("MelBands", "numberBands", 40), sr) {
      // as rhythmtransform uses the mel bands to be initialized, we don't want
      // this plugin to take MelBands as the name
      AlgorithmInfo<standard::Algorithm> info = standard::AlgorithmFactory::getInfo("RhythmTransform");
      setName(info.name);
      setDescription(info.description);
    }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "rhythmtransform";
    d.name = "rhythm transform";
    d.description = "frames in the rhythm domain";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 129; // rhythm transform framesize = 256. thus 256/2+1=129
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleRate = _sampleRate/_stepSize/32; // rhyhtmTransform is configured with a hopsize=32
    d.sampleType = OutputDescriptor::FixedSampleRate;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime ts) {

    computeSpectrum(inputBuffers);

    vector<Real> melBands;
    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(melBands);
    _algo->compute();

    // TODO: name under pool should be unique or otherwise the next time this algo is run
    // will pick the wrong data
    _pool.add("melbands", melBands);

    return FeatureSet();
  }

  FeatureSet getRemainingFeatures() {
    standard::Algorithm * rhythmTransform = standard::AlgorithmFactory::create("RhythmTransform");

    vector<vector<Real> > rhythm;
    FeatureSet result;

    rhythmTransform->input("melBands").set(_pool.value<vector<vector<Real> > >("melbands"));
    rhythmTransform->output("rhythm").set(rhythm);
    rhythmTransform->compute();
    _pool.remove("melbands");

    for (int i=0; i<(int)rhythm.size(); i++) {
      result[0].push_back(makeFeature(rhythm[i]));
    }

    return result;
  }
};

class MFCC : public VampWrapper {
public:

  MFCC(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("MFCC"), sr) {}

  //InputDomain getInputDomain() const { return FrequencyDomain; }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "bands";
    d.name = "Mel Bands Energy";
    d.description = "Mel bands' energy";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = _algo->parameter("numberBands").toInt();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "mfcc";
    d.name = "MFCC";
    d.description = "Mel frequency cepstrum coefficients";
    d.binCount = _algo->parameter("numberCoefficients").toInt();
    d.unit = "";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "numberBands";
    d.name = "numberBands";
    d.description = "the number of mel bands";
    d.unit = "";
    d.minValue = 1; // there's a limitation that nBands > nCoeffs
    d.maxValue = 40;
    d.defaultValue = 40;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "numberCoefficients";
    d.name = "numberCoefficients";
    d.description = "the number of output mel coefficients (usually 13)";
    d.unit = "";
    d.minValue = 1;
    d.maxValue = 13;
    d.defaultValue = 13;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "lowFrequencyBound";
    d.name = "lowFrequencyBound";
    d.description = "the lower bound of the frequency range";
    d.unit = "Hz";
    d.minValue = 0;
    d.maxValue = _sampleRate/2.0-1;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 50;
    list.push_back(d);

    d.identifier = "highFrequencyBound";
    d.name = "highFrequencyBound";
    d.description = "the higher bound of the frequency range";
    d.unit = "Hz";
    d.minValue = 1;
    d.maxValue = _sampleRate/2.0;
    d.defaultValue = 11000;
    d.isQuantized = true;
    d.quantizeStep = 100;
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    return _algo->parameter(id).toReal();
  }

  void setParameter(string id, float value) {
    if (id == "numberBands") {
      if (value <= getParameter("numberCoefficients")) {
        value = getParameter("numberCoefficients") + 1;
      }
    }
    const vector<ParameterDescriptor>& params = getParameterDescriptors();
    ParameterMap parameterMap;
    for (int i=0; i < (int)params.size(); i++) {
      if (params[i].identifier == id) parameterMap.add(id, value);
      else parameterMap.add(params[i].identifier, getParameter(params[i].identifier));
      }
    _algo->configure(parameterMap);
  }


  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    vector<float> bands, mfcc;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(bands);
    _algo->output("mfcc").set(mfcc);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(bands));
    result[1].push_back(makeFeature(mfcc));

    return result;
  }
};

class GFCC : public VampWrapper {
public:

  GFCC(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("GFCC"), sr) {}

  //InputDomain getInputDomain() const { return FrequencyDomain; }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "bands";
    d.name = "ERB Bands Energy";
    d.description = "ERB bands' energy";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = _algo->parameter("numberBands").toInt();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "gfcc";
    d.name = "GFCC";
    d.description = "Gammatone feature cepstrum coefficients";
    d.binCount = _algo->parameter("numberCoefficients").toInt();
    d.unit = "";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "numberBands";
    d.name = "numberBands";
    d.description = "the number of ERB bands";
    d.unit = "";
    d.minValue = 1; // there's a limitation that nBands > nCoeffs
    d.maxValue = 40;
    d.defaultValue = 40;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "numberCoefficients";
    d.name = "numberCoefficients";
    d.description = "the number of output cepstrum coefficients";
    d.unit = "";
    d.minValue = 1;
    d.maxValue = 13;
    d.defaultValue = 13;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "lowFrequencyBound";
    d.name = "lowFrequencyBound";
    d.description = "the lower bound of the frequency range";
    d.unit = "Hz";
    d.minValue = 0;
    d.maxValue = _sampleRate/2.0-1;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 50;
    list.push_back(d);

    d.identifier = "highFrequencyBound";
    d.name = "highFrequencyBound";
    d.description = "the higher bound of the frequency range";
    d.unit = "Hz";
    d.minValue = 1;
    d.maxValue = _sampleRate/2.0;
    d.defaultValue = 11000;
    d.isQuantized = true;
    d.quantizeStep = 100;
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    return _algo->parameter(id).toReal();
  }

  void setParameter(string id, float value) {
    if (id == "numberBands") {
      if (value <= getParameter("numberCoefficients")) {
        value = getParameter("numberCoefficients") + 1;
      }
    }
    const vector<ParameterDescriptor>& params = getParameterDescriptors();
    ParameterMap parameterMap;
    for (int i=0; i < (int)params.size(); i++) {
      if (params[i].identifier == id) parameterMap.add(id, value);
      else parameterMap.add(params[i].identifier, getParameter(params[i].identifier));
      }
    _algo->configure(parameterMap);
  }


  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    vector<float> bands, mfcc;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(bands);
    _algo->output("gfcc").set(mfcc);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(bands));
    result[1].push_back(makeFeature(mfcc));

    return result;
  }
};

class SBic : public VampWrapper  {
public:

  SBic(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("MFCC"), sr) {
      // as SBic uses the MFCC algo to be initialized, we don't want
      // this plugin to take MFCC as the name nor its description
      AlgorithmInfo<standard::Algorithm> info = standard::AlgorithmFactory::getInfo("SBic");
      setName(info.name);
      setDescription(info.description);
    }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "sbic";
    d.name = "sbic";
    d.description = "bayesian information criterion segmenter";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 0;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleRate = _sampleRate;
    d.sampleType = OutputDescriptor::VariableSampleRate;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime ts) {

    computeSpectrum(inputBuffers);

    vector<float> bands, mfcc;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(bands); // not used at all
    _algo->output("mfcc").set(mfcc);

    _algo->compute();

    _pool.add("mfcc", mfcc);

    return FeatureSet();
  }

  FeatureSet getRemainingFeatures() {
    standard::Algorithm * SBic =
      standard::AlgorithmFactory::create("SBic", "minLength", 10,
                                         "size1", 1000, "inc1", 300,
                                         "size2", 600, "inc2", 50, "cpw", 5);

    FeatureSet result;

    TNT::Array2D<Real> mfcc(transpose(vecvecToArray2D(_pool.value<vector<vector<Real> > >("mfcc"))));
    vector<Real> segments;

    SBic->input("features").set(mfcc);
    SBic->output("segmentation").set(segments);
    SBic->compute();
    _pool.remove("mfcc");
    delete SBic;

    for (int i=0; i<(int)segments.size(); i++) {
      Feature t;
      t.hasTimestamp = true;
      //t.timestamp = Vamp::RealTime::frame2RealTime(long(segments[i]), size_t(_sampleRate));
      t.timestamp = Vamp::RealTime::fromSeconds(segments[i]/_sampleRate*_stepSize);
      result[0].push_back(t);
    }

    return result;
  }
};

class LPC : public VampWrapper  {
public:

  LPC(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("LPC"), sr) {}

  InputDomain getInputDomain() const { return TimeDomain; }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "lpc";
    d.name = "LPC";
    d.description = "LPC coeffs";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = _algo->parameter("order").toInt() + 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "lpcreflection";
    d.name = "LPC reflection";
    d.description = "LPC reflection";
    d.binCount = _algo->parameter("order").toInt();
    d.unit = "";
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "warped";
    d.name = "type";
    d.description = "use regular or warped lpc";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 1;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("regular");
    d.valueNames.push_back("warped");
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    if (_algo->parameter("type") == "regular") return 0;
    else return 1;
  }

  void setParameter(string id, float value) {
    if (int(value) == 0) _algo->configure("type", "regular");
    else _algo->configure("type", "warped");
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    RogueVector<float> inputr(const_cast<float*>(inputBuffers[0]), _blockSize);
    vector<float>& input = static_cast<vector<float>&>(inputr);

    vector<float> LPC, LPCR;

    _algo->input("frame").set(input);
    _algo->output("lpc").set(LPC);
    _algo->output("reflection").set(LPCR);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(LPC));
    result[1].push_back(makeFeature(LPCR));

    return result;
  }
};


class Friction : public VampWrapper  {
public:

  Friction(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("Tristimulus"), sr) {}

  string getIdentifier() const { return "friction"; }
  string getName() const { return "Friction"; }

  OutputList getOutputDescriptors() const {
    return genericDescriptor("", 1);
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computePeaks(inputBuffers);

    vector<float> value(3);

    _algo->input("magnitudes").set(_peakmags);
    _algo->input("frequencies").set(_peakfreqs);
    _algo->output(_algo->outputNames()[0]).set(value);

    _algo->compute();

    float friction = 2. - (value[1]-value[2]);
    //friction = min(max(friction, 0.f), 2.0f);
    return returnFeature(friction);
  }
};


class Crunchiness : public VampWrapper  {
protected:
  deque<float> mem;
public:

  Crunchiness(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("PitchSalience"), sr) {
    mem = deque<float>(4, 0.0);
  }

  string getIdentifier() const { return "crunchiness"; }
  string getName() const { return "Crunchiness"; }

  OutputList getOutputDescriptors() const {
    return genericDescriptor("", 1);
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    float salience;

    _algo->input("spectrum").set(_spectrum);
    _algo->output(_algo->outputNames()[0]).set(salience);

    _algo->compute();

    mem.pop_front();
    mem.push_back(salience);

    vector<float> v(mem.begin(), mem.end());
    float m = mean(v);
    float std = stddev(v, m);

    return returnFeature(m+std);
  }
};


class Perculleys : public VampWrapper  {
protected:
  deque<float> mem;
public:

  Perculleys(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("SpectralContrast"), sr) {
  }

  string getIdentifier() const { return "perculleys"; }
  string getName() const { return "PercussiveValleys"; }

  OutputList getOutputDescriptors() const {
    return genericDescriptor("", 2);
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);

    vector<float> contrast, valleys;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("spectralContrast").set(contrast);
    _algo->output("spectralValley").set(valleys);

    _algo->compute();

    float m = mean(valleys);

    // linear fit
    float _range = 1.0;
    float scaler = _range / (valleys.size() - 1.0);

    float mean_x = _range / 2.0;
    float mean_y = m;

    float ss_xx = 0.0;
    float ss_xy = 0.0;
    for (int i=0; i<int(valleys.size()); ++i) {
      float tmp = float(i) * scaler - mean_x;
      ss_xx += tmp * tmp;
      ss_xy += tmp * (valleys[i] - mean_y);
    }

    float decrease = ss_xy / ss_xx;

    vector<float> pclys(2);

    pclys[0] = m;
    pclys[1] = decrease;

    return returnFeature(pclys);
  }
};

class HPCP : public VampWrapper  {
public:

  HPCP(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("HPCP"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "hpcp";
    d.name = "HPCP";
    d.description = "Harmonic Pitch Class Profile";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = _algo->parameter("size").toInt();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computePeaks(inputBuffers);

    vector<float> hpcp;

    _algo->input("frequencies").set(_peakfreqs);
    _algo->input("magnitudes").set(_peakmags);
    _algo->output("hpcp").set(hpcp);

    _algo->compute();

    FeatureSet result;
    result[0].push_back(makeFeature(hpcp));

    return result;
  }
};

class SpectralWhitening : public VampWrapper  {
public:

  SpectralWhitening(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("SpectralWhitening"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "magnitudes";
    d.name = "magnitudes";
    d.description = "spectral magnitudes whitened";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 100; // limitation from spectralPeaks maxnumber of peaks _spectrum.size();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    return ParameterList();
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computePeaks(inputBuffers);

    vector<float> mags;

    _algo->input("spectrum").set(_spectrum);
    _algo->input("frequencies").set(_peakfreqs);
    _algo->input("magnitudes").set(_peakmags);
    _algo->output("magnitudes").set(mags);

    _algo->compute();


    FeatureSet result;
    result[0].push_back(makeFeature(mags));

    return result;
  }
};

class SpectralPeaks: public VampWrapper  {
public:

  SpectralPeaks(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("SpectralPeaks"), sr) {}

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "frequencies";
    d.name = "frequencies";
    d.description = "spectral peaks' frequencies";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 100; // limitation from spectralPeaks maxnumber of peaks _spectrum.size();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    d.identifier = "magnitudes";
    d.name = "magnitudes";
    d.description = "spectral peaks' magnitudes";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 100; // limitation from spectralPeaks maxnumber of peaks _spectrum.size();
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::OneSamplePerStep;
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;

    d.identifier = "maxPeaks";
    d.name = "maxPeaks";
    d.description = "the maximum number of peaks to be found";
    d.unit = "";
    d.minValue = 1;
    d.maxValue = 100;
    d.defaultValue = 100;
    d.isQuantized = true;
    d.quantizeStep = 1;
    list.push_back(d);

    d.identifier = "maxFrequency";
    d.name = "maxFrequency";
    d.description = "the higher frequency bound";
    d.unit = "Hz";
    d.minValue = 1;
    d.maxValue = _sampleRate/2.0;
    d.defaultValue = 5000;
    d.isQuantized = true;
    d.quantizeStep = 100;
    list.push_back(d);

    d.identifier = "minFrequency";
    d.name = "minFrequency";
    d.description = "the lower frequency bound";
    d.unit = "Hz";
    d.minValue = 0;
    d.maxValue = _sampleRate/2.0;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 50;
    list.push_back(d);

    d.identifier = "magnitudeThreshold";
    d.name = "magnitudeThreshold";
    d.description = "ignore peaks below this given threshold";
    d.unit = "";
    d.minValue = 0;
    d.defaultValue = 0;
    list.push_back(d);

    d.identifier = "orderBy";
    d.name = "orderBy";
    d.description = "peaks' ordering type";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 1;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.valueNames.push_back("frequency");
    d.valueNames.push_back("magnitude");
    list.push_back(d);

    return list;
  }

  float getParameter(string id) const {
    if (id == "orderBy") {
      if (_algo->parameter("orderBy") == "frequency") return 0;
      else return 1;
    }
    else return _algo->parameter(id).toReal();
  }

  void setParameter(string id, float value) {
    ParameterMap params;
    if (id == "orderBy") {
      if (value == 0.0) params.add("orderBy", "frequency");
      else _algo->configure("orderBy", "magnitude");
    }
    else params.add(id, value);
    _algo->configure(params);
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computePeaks(inputBuffers);

    vector<float> mags, freqs;

    _algo->input("spectrum").set(_spectrum);
    _algo->output("frequencies").set(freqs);
    _algo->output("magnitudes").set(mags);

    _algo->compute();


    FeatureSet result;
    result[0].push_back(makeFeature(freqs));
    result[1].push_back(makeFeature(mags));

    return result;
  }
};

class NoveltyCurve : public VampWrapper  {
    standard::Algorithm * _noveltyCurve;
    std::string _weightCurve, _bandsType;

public:
  NoveltyCurve(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("FrequencyBands"), sr) {
      // as NoveltyCurve uses BarkBands algo to be initialized, we don't want
      // this plugin to take BarkBands as the name nor its description
      _noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve");
      _weightCurve = "flat";
      _bandsType = "bark";
      AlgorithmInfo<standard::Algorithm> info = standard::AlgorithmFactory::getInfo("NoveltyCurve");
      setName(info.name);
      setDescription(info.description);
    }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "noveltyCurve";
    d.name = "noveltyCurve";
    d.description = "the novelty curve of the frequency bands";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleRate = _sampleRate;
    d.sampleType = OutputDescriptor::VariableSampleRate;
    list.push_back(d);
    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "bandsType";
    d.name = "bands type";
    d.description = "the type of bands";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 0;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("bark");
    d.valueNames.push_back("scheirer");
    d.valueNames.push_back("reduced");
    list.push_back(d);

    d.valueNames.clear();
    d.identifier = "weightCurveType";
    d.name = "weight curve";
    d.description = "the type of curve to weight the bands";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 0;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("flat");
    d.valueNames.push_back("triangle");
    d.valueNames.push_back("inverse_triangle");
    d.valueNames.push_back("parabola");
    d.valueNames.push_back("inverse_parabola");
    d.valueNames.push_back("linear");
    d.valueNames.push_back("quadratic");
    d.valueNames.push_back("inverse_quadratic");

    list.push_back(d);
    return list;
   }

  float getParameter(string id) const {
    if (id == "weightCurveType") {
      if (_weightCurve == "flat")              return 0;
      if (_weightCurve == "triangle")          return 1;
      if (_weightCurve == "inverse_triangle")  return 2;
      if (_weightCurve == "parabola")          return 3;
      if (_weightCurve == "inverse_parabola")  return 4;
      if (_weightCurve == "linear")            return 5;
      if (_weightCurve == "quadratic")         return 6;
      if (_weightCurve == "inverse_quadratic") return 7;
    }
    if (id == "bandsType") {
      if (_bandsType == "bark")     return 0;
      if (_bandsType == "scheirer") return 1;
      if (_bandsType == "reduced")  return 2;
    }
    return 0;
  }

  void setParameter(string id, float value) {
    if (id=="weightCurveType") {
      switch(int(value)) {
        case 0: _weightCurve = "flat"; break;
        case 1: _weightCurve = "triangle"; break;
        case 2: _weightCurve = "inverse_triangle"; break;
        case 3: _weightCurve = "parabola"; break;
        case 4: _weightCurve = "inverse_parabola"; break;
        case 5: _weightCurve = "linear"; break;
        case 6: _weightCurve = "quadratic"; break;
        case 7: _weightCurve = "inverse_quadratic"; break;
        default : _weightCurve = "flat";
      }
    }
    if (id=="bandsType") {
      const Real scheirerBands[] = { 0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 22050.0 };
      const Real reducedBands[]  = { 0.0, 50.0, 100.0, 150.0, 200.0 };
      const Real barkBands[]     = { 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0,
                                     920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
                                     3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
                                     15500.0, 20500.0, 27000.0 };
      std::vector<Real> freqBands;

      // if 0: by default frequencyBands is configured with bark bands
      switch(int(value))  {
        case 0:
          freqBands = arrayToVector<Real>(barkBands);
          _bandsType = "bark";
          break;
        case 1:
          freqBands = arrayToVector<Real>(scheirerBands);
          _bandsType = "scheirer";
          break;
        case 2:
          freqBands = arrayToVector<Real>(reducedBands);
          _bandsType = "reduced";
          break;
        default:
          freqBands = arrayToVector<Real>(barkBands);
          _bandsType = "bark";
          break;
      }
      _algo->configure("sampleRate", _sampleRate, "frequencyBands", freqBands);
    }
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);
    vector<float> bands;
    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(bands);
    _algo->compute();
    _pool.add("bands", bands);
    return FeatureSet();
  }

  FeatureSet getRemainingFeatures() {
    FeatureSet result;
    vector<Real> novelty;
    _noveltyCurve->configure("frameRate", float(_sampleRate)/float(_stepSize),
                             "weightCurveType", _weightCurve);
    _noveltyCurve->input("frequencyBands").set(_pool.value<vector<vector<Real> > >("bands"));
    _noveltyCurve->output("novelty").set(novelty);
    _noveltyCurve->compute();
    _pool.remove("bands");

    Real stepTime = _stepSize/_sampleRate;

    for (int i=0; i<(int)novelty.size(); i++) {
      Feature f;
      f.hasTimestamp=true;
      f.timestamp = Vamp::RealTime::fromSeconds(i*stepTime);
      f.values.push_back(novelty[i]);
      result[0].push_back(f);
    }

    return result;
  }
};

class BpmHistogram : public VampWrapper  {
    standard::Algorithm * _noveltyCurve;
    std::string _weightCurve, _bandsType;
    //Real _maxBpm; _minBpm;

public:
  BpmHistogram(float sr) :
    VampWrapper(standard::AlgorithmFactory::create("FrequencyBands"), sr) {
      // as NoveltyCurve uses BarkBands algo to be initialized, we don't want
      // this plugin to take BarkBands as the name nor its description
      _noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve");
      _weightCurve = "flat";
      _bandsType = "bark";
      AlgorithmInfo<streaming::Algorithm> info = streaming::AlgorithmFactory::getInfo("BpmHistogram");
      setName(info.name);
      setDescription(info.description);
    }

  OutputList getOutputDescriptors() const {
    OutputList list;

    OutputDescriptor d;
    d.identifier = "sinusoid";
    d.name = "sinusoid";
    d.description = "the sinusoidal model of the ticks";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = false;
    d.isQuantized = false;
    d.sampleRate = _sampleRate;
    d.sampleType = OutputDescriptor::VariableSampleRate;

    list.push_back(d);

    d.identifier = "tempogram";
    d.name = "tempogram";
    d.description = "bpm evolution";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 561; // fixed by the default values of bpmHistogram (i.e. maxbpm)
    d.hasKnownExtents = false;
    d.sampleRate = _sampleRate/_stepSize/32; // fixed by overlap param in bpmHistogram
    d.sampleType = OutputDescriptor::FixedSampleRate; // do not change it!!
    list.push_back(d);

    d.identifier = "ticks";
    d.name = "ticks";
    d.description = "tick positions";
    d.unit = "";
    d.hasFixedBinCount = true; //false;
    d.binCount = 0;
    d.hasKnownExtents = false;
    d.sampleRate = _sampleRate;
    d.sampleType = OutputDescriptor::VariableSampleRate;
    list.push_back(d);

    return list;
  }

  ParameterList getParameterDescriptors() const {
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "bandsType";
    d.name = "bands type";
    d.description = "the type of bands";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 0;
    d.defaultValue = 0;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames.push_back("bark");
    d.valueNames.push_back("scheirer");
    d.valueNames.push_back("reduced");
    list.push_back(d);

    //d.valueNames.clear();
    //d.identifier = "weightCurve";
    //d.name = "weight curve";
    //d.description = "the type of curve to weight the bands";
    //d.unit = "";
    //d.minValue = 0;
    //d.maxValue = 0;
    //d.defaultValue = 0;
    //d.isQuantized = true;
    //d.quantizeStep = 1;
    //d.valueNames.push_back("flat");
    //d.valueNames.push_back("triangle");
    //d.valueNames.push_back("inverse_triangle");
    //d.valueNames.push_back("parabola");
    //d.valueNames.push_back("inverse_parabola");
    //d.valueNames.push_back("linear");
    //d.valueNames.push_back("quadratic");
    //d.valueNames.push_back("inverse_quadratic");

    list.push_back(d);
    return list;
   }

  float getParameter(string id) const {
    if (id == "weightCurveType") {
      if (_weightCurve == "flat")              return 0;
      if (_weightCurve == "triangle")          return 1;
      if (_weightCurve == "inverse_triangle")  return 2;
      if (_weightCurve == "parabola")          return 3;
      if (_weightCurve == "inverse_parabola")  return 4;
      if (_weightCurve == "linear")            return 5;
      if (_weightCurve == "quadratic")         return 6;
      if (_weightCurve == "inverse_quadratic") return 7;
    }
    if (id == "bandsType") {
      if (_bandsType == "bark")     return 0;
      if (_bandsType == "scheirer") return 1;
      if (_bandsType == "reduced")  return 2;
    }
    return 0;
  }


  void setParameter(string id, float value) {
    if (id=="weightCurveType") {
      switch(int(value)) {
        case 0: _weightCurve = "flat"; break;
        case 1: _weightCurve = "triangle"; break;
        case 2: _weightCurve = "inverse_triangle"; break;
        case 3: _weightCurve = "parabola"; break;
        case 4: _weightCurve = "inverse_parabola"; break;
        case 5: _weightCurve = "linear"; break;
        case 6: _weightCurve = "quadratic"; break;
        case 7: _weightCurve = "inverse_quadratic"; break;
        default : _weightCurve = "flat";
      }
    }
    if (id=="bandsType") {
      const Real scheirerBands[] = { 0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 22050.0 };
      const Real reducedBands[]  = { 0.0, 50.0, 100.0, 150.0, 200.0 };
      const Real barkBands[]     = { 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0,
                                     920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
                                     3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
                                     15500.0, 20500.0, 27000.0 };
      std::vector<Real> freqBands;

      // if 0: by default frequencyBands is configured with bark bands
      switch(int(value))  {
        case 0:
          freqBands = arrayToVector<Real>(barkBands);
          _bandsType = "bark";
          break;
        case 1:
          freqBands = arrayToVector<Real>(scheirerBands);
          _bandsType = "scheirer";
          break;
        case 2:
          freqBands = arrayToVector<Real>(reducedBands);
          _bandsType = "reduced";
          break;
        default:
          freqBands = arrayToVector<Real>(barkBands);
          _bandsType = "bark";
          break;
      }
      _algo->configure("sampleRate", _sampleRate, "frequencyBands", freqBands);
    }
  }

  FeatureSet process(const float *const *inputBuffers, Vamp::RealTime) {
    computeSpectrum(inputBuffers);
    vector<float> bands;
    _algo->input("spectrum").set(_spectrum);
    _algo->output("bands").set(bands);
    _algo->compute();
    _pool.add("bands", bands);
    return FeatureSet();
  }

  FeatureSet getRemainingFeatures() {

    FeatureSet result;
    vector<Real> novelty;
    _noveltyCurve->configure("frameRate", float(_sampleRate)/float(_stepSize),
                             "weightCurveType", _weightCurve);
    _noveltyCurve->input("frequencyBands").set(_pool.value<vector<vector<Real> > >("bands"));
    _noveltyCurve->output("novelty").set(novelty);
    _noveltyCurve->compute();
    _pool.remove("bands");

    streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
    streaming::Algorithm* gen = new streaming::VectorInput<Real>(&novelty);
    streaming::Algorithm * bpmHist = factory.create("BpmHistogram",
                                                    "zeroPadding", 1,
                                                    "frameRate", Real(_sampleRate)/Real(_stepSize));

    connect(gen->output("data"), bpmHist->input("novelty"));
    connect(bpmHist->output("bpm"),          _pool, "bpm");
    connect(bpmHist->output("bpmCandidates"), _pool, "bpmCandidates");
    connect(bpmHist->output("bpmMagnitudes"), _pool, "bpmMagnitudes");
    connect(bpmHist->output("tempogram"), _pool, "tempogram");
    connect(bpmHist->output("frameBpms"), _pool, "frameBpms");
    connect(bpmHist->output("ticks"),       _pool, "ticks");
    connect(bpmHist->output("ticksMagnitude"), _pool, "ticksMagnitude");
    connect(bpmHist->output("sinusoid"),    _pool, "sinusoid");
    Network network(gen);
    network.run();    // runGenerator(gen);
    network.clear();  // deleteNetwork(gen);

    _pool.remove("bpm");
    _pool.remove("bpmCandidates");
    _pool.remove("bpmMagnitudes");
    _pool.remove("frameBpms");
    //_pool.remove("ticks");
    _pool.remove("ticksMagnitude");

    Real stepTime = Real(_stepSize)/Real(_sampleRate);
    const vector<Real> & sinusoid = _pool.value<vector<Real> >("sinusoid");
    const vector<Real> & ticks = _pool.value<vector<Real> > ("ticks");
    const TNT::Array2D<Real>& tempogramMatrix = _pool.value<vector<TNT::Array2D<Real> > >("tempogram")[0];
    const vector<vector<Real> > tempogram = array2DToVecvec(tempogramMatrix);
    int tempogramSize = tempogram.size();

    for (int i=0; i<(int)sinusoid.size(); i++) {
      Feature f;
      f.hasTimestamp=true;
      f.timestamp = Vamp::RealTime::fromSeconds(i*stepTime);
      f.values.push_back(sinusoid[i]);
      result[0].push_back(f);
      if (i>=tempogramSize) continue;
      result[1].push_back(makeFeature(tempogram[i]));
    }
    vector<Real>::const_iterator it = ticks.begin();
    while (it != ticks.end()) {
      Feature f;
      f.hasTimestamp=true;
      f.timestamp = Vamp::RealTime::fromSeconds(*it);
      result[2].push_back(f);
      ++it;
    }

    return result;
  }
};
