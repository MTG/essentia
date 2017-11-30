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

#include "multipitchmelodia.h"

using namespace std;

namespace essentia {
namespace standard {


const char* MultiPitchMelodia::name = "MultiPitchMelodia";
const char* MultiPitchMelodia::category = "Pitch";
const char* MultiPitchMelodia::description = DOC("This algorithm estimates multiple fundamental frequency contours from an audio signal. It is a multi pitch version of the MELODIA algorithm described in [1]. While the algorithm is originally designed to extract melody in polyphonic music, this implementation is adapted for multiple sources. The approach is based on the creation and characterization of pitch contours, time continuous sequences of pitch candidates grouped using auditory streaming cues. To this end, PitchSalienceFunction, PitchSalienceFunctionPeaks, PitchContours, and PitchContoursMonoMelody algorithms are employed. It is strongly advised to use the default parameter values which are optimized according to [1] (where further details are provided) except for minFrequency, maxFrequency, and voicingTolerance, which will depend on your application.\n"
"\n"
"The output is a vector of estimated melody pitch values and a vector of confidence values.\n"
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n\n"
"  [2] http://mtg.upf.edu/technologies/melodia\n\n"
"  [3] http://www.justinsalamon.com/melody-extraction\n"
);

void MultiPitchMelodia::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  string windowType = "hann";
  int zeroPaddingFactor = 4;
  int maxSpectralPeaks = 100;

  Real referenceFrequency = parameter("referenceFrequency").toReal();
  Real binResolution = parameter("binResolution").toReal();
  Real magnitudeThreshold = parameter("magnitudeThreshold").toReal();
  Real magnitudeCompression = parameter("magnitudeCompression").toReal();
  int numberHarmonics = parameter("numberHarmonics").toInt();
  Real harmonicWeight = parameter("harmonicWeight").toReal();

  Real minFrequency = parameter("minFrequency").toReal();
  Real maxFrequency = parameter("maxFrequency").toReal();

  Real peakFrameThreshold = parameter("peakFrameThreshold").toReal();
  Real peakDistributionThreshold = parameter("peakDistributionThreshold").toReal();
  Real pitchContinuity = parameter("pitchContinuity").toReal();
  Real timeContinuity = parameter("timeContinuity").toReal();
  Real minDuration = parameter("minDuration").toReal();

  int filterIterations = parameter("filterIterations").toInt();
  bool guessUnvoiced = parameter("guessUnvoiced").toBool();

  // Pre-processing
  _frameCutter->configure("frameSize", frameSize,
                           "hopSize", hopSize,
                           "startFromZero", false);

  _windowing->configure("size", frameSize,
                        "zeroPadding", (zeroPaddingFactor-1) * frameSize,
                        "type", windowType);
  // Spectral peaks
  _spectrum->configure("size", frameSize * zeroPaddingFactor);


  _spectralPeaks->configure(
                            "minFrequency", 1,  // to avoid zero frequencies
                            "maxFrequency", 20000,
                            "maxPeaks", maxSpectralPeaks,
                            "sampleRate", sampleRate,
                            "magnitudeThreshold", 0,
                            "orderBy", "magnitude");

  // Pitch salience contours
  _pitchSalienceFunction->configure("binResolution", binResolution,
                                    "referenceFrequency", referenceFrequency,
                                    "magnitudeThreshold", magnitudeThreshold,
                                    "magnitudeCompression", magnitudeCompression,
                                    "numberHarmonics", numberHarmonics,
                                    "harmonicWeight", harmonicWeight);

  // exaggerated min/max values to take all peaks
  // independend of the range of salience function
  _pitchSalienceFunctionPeaks->configure("binResolution", binResolution,
                                         "referenceFrequency", referenceFrequency,
                                         "minFrequency", 1,
                                         "maxFrequency", 20000);

  // NOTE: previously used hardcoded values 
  // - peakFrameThreshold = 0.5
  // - peakDistributionThreshold = 2.0 
  // Are those good to use as default values? 
  _pitchContours->configure("sampleRate", sampleRate,
                                            "hopSize", hopSize,
                                            "binResolution", binResolution,
                                            "peakFrameThreshold", peakFrameThreshold,
                                            "peakDistributionThreshold", peakDistributionThreshold,
                                            "pitchContinuity", pitchContinuity,
                                            "timeContinuity", timeContinuity,
                                            "minDuration", minDuration);

  // Melody detection
  _pitchContoursMelody->configure("referenceFrequency", referenceFrequency,
                                          "binResolution", binResolution,
                                          "sampleRate", sampleRate,
                                          "hopSize", hopSize,
                                          "filterIterations", filterIterations,
                                          "guessUnvoiced", guessUnvoiced,
                                          "minFrequency", minFrequency,
                                          "maxFrequency", maxFrequency);
}

void MultiPitchMelodia::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& pitch = _pitch.get();
  if (signal.empty()) {
    pitch.clear();
    return;
  }

  // Pre-processing
  vector<Real> frame;
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  vector<Real> frameWindowed;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(frameWindowed);

  // Spectral peaks
  vector<Real> frameSpectrum;
  _spectrum->input("frame").set(frameWindowed);
  _spectrum->output("spectrum").set(frameSpectrum);

  vector<Real> frameFrequencies;
  vector<Real> frameMagnitudes;
  _spectralPeaks->input("spectrum").set(frameSpectrum);
  _spectralPeaks->output("frequencies").set(frameFrequencies);
  _spectralPeaks->output("magnitudes").set(frameMagnitudes);

  // Pitch salience contours
  vector<Real> frameSalience;
  _pitchSalienceFunction->input("frequencies").set(frameFrequencies);
  _pitchSalienceFunction->input("magnitudes").set(frameMagnitudes);
  _pitchSalienceFunction->output("salienceFunction").set(frameSalience);

  vector<Real> frameSalienceBins;
  vector<Real> frameSalienceValues;
  _pitchSalienceFunctionPeaks->input("salienceFunction").set(frameSalience);
  _pitchSalienceFunctionPeaks->output("salienceBins").set(frameSalienceBins);
  _pitchSalienceFunctionPeaks->output("salienceValues").set(frameSalienceValues);


  vector<vector<Real> > peakBins;
  vector<vector<Real> > peakSaliences;

  while (true) {
    // get a frame
    _frameCutter->compute();

    if (!frame.size()) {
      break;
    }

    _windowing->compute();

    // calculate spectrum
    _spectrum->compute();

    // calculate spectral peaks
    _spectralPeaks->compute();

    // calculate salience function
    _pitchSalienceFunction->compute();

    // calculate peaks of salience function
    _pitchSalienceFunctionPeaks->compute();

    peakBins.push_back(frameSalienceBins);
    peakSaliences.push_back(frameSalienceValues);
  }

  // calculate pitch contours
  vector<vector<Real> > contoursBins;
  vector<vector<Real> > contoursSaliences;
  vector<Real> contoursStartTimes;
  Real duration;

  _pitchContours->input("peakBins").set(peakBins);
  _pitchContours->input("peakSaliences").set(peakSaliences);
  _pitchContours->output("contoursBins").set(contoursBins);
  _pitchContours->output("contoursSaliences").set(contoursSaliences);
  _pitchContours->output("contoursStartTimes").set(contoursStartTimes);
  _pitchContours->output("duration").set(duration);

  _pitchContours->compute();

  // calculate melody
  _pitchContoursMelody->input("contoursBins").set(contoursBins);
  _pitchContoursMelody->input("contoursSaliences").set(contoursSaliences);
  _pitchContoursMelody->input("contoursStartTimes").set(contoursStartTimes);
  _pitchContoursMelody->input("duration").set(duration);
  _pitchContoursMelody->output("pitch").set(pitch);
  _pitchContoursMelody->compute();
}

MultiPitchMelodia::~MultiPitchMelodia() {
    // Pre-processing
    delete _frameCutter;
    delete _windowing;

    // Spectral peaks
    delete _spectrum;
    delete _spectralPeaks;

    // Pitch salience contours
    delete _pitchSalienceFunction;
    delete _pitchSalienceFunctionPeaks;
    delete _pitchContours;

    // Melody
    delete _pitchContoursMelody;
}


} // namespace standard
} // namespace essentia

#include "poolstorage.h"

namespace essentia {
namespace streaming {


MultiPitchMelodia::MultiPitchMelodia() : AlgorithmComposite() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _frameCutter                = factory.create("FrameCutter");
  _windowing                  = factory.create("Windowing");
  _spectrum                   = factory.create("Spectrum");
  _spectralPeaks              = factory.create("SpectralPeaks");
  _pitchSalienceFunction      = factory.create("PitchSalienceFunction");
  _pitchSalienceFunctionPeaks = factory.create("PitchSalienceFunctionPeaks");

  _pitchContours = standard::AlgorithmFactory::create("PitchContours");
  _pitchContoursMelody = standard::AlgorithmFactory::create("PitchContoursMelody");

  // TODO delete
  //_poolStorageBins = new PoolStorage<vector<vector<Real> > >(&_pool, "internal.saliencebins");
  //_poolStorageValues = new PoolStorage<vector<vector<Real> > >(&_pool, "internal.saliencevalues");

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_pitch, "pitch", "the estimated pitch values per frames [Hz]");

  // Connect input proxy
  _signal >> _frameCutter->input("signal");

  // Connect frame-wize algorithms
  _frameCutter->output("frame")   >> _windowing->input("frame");
  _windowing->output("frame")     >> _spectrum->input("frame");
  _spectrum->output("spectrum")   >> _spectralPeaks->input("spectrum");

  _spectralPeaks->output("frequencies")                 >> _pitchSalienceFunction->input("frequencies");
  _spectralPeaks->output("magnitudes")                  >> _pitchSalienceFunction->input("magnitudes");
  _pitchSalienceFunction->output("salienceFunction")    >> _pitchSalienceFunctionPeaks->input("salienceFunction");
  _pitchSalienceFunctionPeaks->output("salienceBins")   >> PC(_pool, "internal.saliencebins");
  _pitchSalienceFunctionPeaks->output("salienceValues") >> PC(_pool, "internal.saliencevalues");

  //_network = new scheduler::Network(_frameCutter);

  // Connect track-wise algorithms // TODO It is yet not possible to stream
  // from a Pool, but it would be a nice option to add in the future. Currently,
  // we have to use standard mode for post-processing.

  //_poolStorageBins->output("data")              >> _pitchContours->input("peakBins");
  //_poolStorageValues->output("data")            >> _pitchContours->input("peakSaliences");
  //_pitchContours->output("contoursBins")        >> _pitchContoursMelody->input("contoursBins");
  //_pitchContours->output("contoursSaliences")   >> _pitchContoursMelody->input("contoursSaliences");
  //_pitchContours->output("contoursStartTimes")  >> _pitchContoursMelody->input("contoursStartTimes");
  //_pitchContours->output("duration")            >> _pitchContoursMelody->input("duration");

  // Connect output proxies
  //_pitchContoursMelody->output("pitch") >> _pitch;
  //_pitchContoursMelody->output("pitchConfidence") >> _pitchConfidence;
}

MultiPitchMelodia::~MultiPitchMelodia() {
    //delete _network;
  delete _pitchContours;
  delete _pitchContoursMelody;
}

void MultiPitchMelodia::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  string windowType = "hann";
  int zeroPaddingFactor = 4;
  int maxSpectralPeaks = 100;

  Real referenceFrequency = parameter("referenceFrequency").toReal();
  Real binResolution = parameter("binResolution").toReal();
  Real magnitudeThreshold = parameter("magnitudeThreshold").toReal();
  Real magnitudeCompression = parameter("magnitudeCompression").toReal();
  int numberHarmonics = parameter("numberHarmonics").toInt();
  Real harmonicWeight = parameter("harmonicWeight").toReal();

  Real minFrequency = parameter("minFrequency").toReal();
  Real maxFrequency = parameter("maxFrequency").toReal();

  Real peakFrameThreshold = parameter("peakFrameThreshold").toReal();
  Real peakDistributionThreshold = parameter("peakDistributionThreshold").toReal();
  Real pitchContinuity = parameter("pitchContinuity").toReal();
  Real timeContinuity = parameter("timeContinuity").toReal();
  Real minDuration = parameter("minDuration").toReal();

  //Real voicingTolerance = parameter("voicingTolerance").toReal();
  int filterIterations = parameter("filterIterations").toInt();
  //bool voiceVibrato = parameter("voiceVibrato").toBool();
  bool guessUnvoiced = parameter("guessUnvoiced").toBool();

  // Pre-processing
  _frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "startFromZero", false);

  _windowing->configure("size", frameSize,
                        "zeroPadding", (zeroPaddingFactor-1) * frameSize,
                        "type", windowType);
  // Spectral peaks
  _spectrum->configure("size", frameSize * zeroPaddingFactor);

  _spectralPeaks->configure("minFrequency", 1,
                            "maxFrequency", 20000,
                            "maxPeaks", maxSpectralPeaks,
                            "sampleRate", sampleRate,
                            "magnitudeThreshold", 0,
                            "orderBy", "magnitude");

  // Pitch salience contours
  _pitchSalienceFunction->configure("binResolution", binResolution,
                                    "referenceFrequency", referenceFrequency,
                                    "magnitudeThreshold", magnitudeThreshold,
                                    "magnitudeCompression", magnitudeCompression,
                                    "numberHarmonics", numberHarmonics,
                                    "harmonicWeight", harmonicWeight);

  // exaggerated min/max values to take all peaks
  // independend of the range of salience function
  _pitchSalienceFunctionPeaks->configure("binResolution", binResolution,
                                         "referenceFrequency", referenceFrequency,
                                         "minFrequency", 1,
                                         "maxFrequency", 20000);

  _pitchContours->configure("sampleRate", sampleRate,
                            "hopSize", hopSize,
                            "binResolution", binResolution,
                            "peakFrameThreshold", peakFrameThreshold,
                            "peakDistributionThreshold", peakDistributionThreshold,
                            "pitchContinuity", pitchContinuity,
                            "timeContinuity", timeContinuity,
                            "minDuration", minDuration);

  // Melody detection
  _pitchContoursMelody->configure("referenceFrequency", referenceFrequency,
                                  "binResolution", binResolution,
                                  "sampleRate", sampleRate,
                                  "hopSize", hopSize,
                                  "filterIterations", filterIterations,
                                  "guessUnvoiced", guessUnvoiced,
                                  "minFrequency", minFrequency,
                                  "maxFrequency", maxFrequency);
}

AlgorithmStatus MultiPitchMelodia::process() {
  if (!shouldStop()) return PASS;

  const vector<vector<Real> >& salienceBins = _pool.value<vector<vector<Real> > >("internal.saliencebins");
  const vector<vector<Real> >& salienceValues = _pool.value<vector<vector<Real> > >("internal.saliencevalues");

  // compute pitch contours
  vector<vector<Real> > contoursBins;
  vector<vector<Real> > contoursSaliences;
  vector<Real> contoursStartTimes;
  Real duration;

  _pitchContours->input("peakBins").set(salienceBins);
  _pitchContours->input("peakSaliences").set(salienceValues);
  _pitchContours->output("contoursBins").set(contoursBins);
  _pitchContours->output("contoursSaliences").set(contoursSaliences);
  _pitchContours->output("contoursStartTimes").set(contoursStartTimes);
  _pitchContours->output("duration").set(duration);
  _pitchContours->compute();

  // compute melody
  vector<vector<Real> > pitch;
  _pitchContoursMelody->input("contoursBins").set(contoursBins);
  _pitchContoursMelody->input("contoursSaliences").set(contoursSaliences);
  _pitchContoursMelody->input("contoursStartTimes").set(contoursStartTimes);
  _pitchContoursMelody->input("duration").set(duration);
  _pitchContoursMelody->output("pitch").set(pitch);
  _pitchContoursMelody->compute();

  _pitch.push(pitch);

  return FINISHED;
}

void MultiPitchMelodia::reset() {
  AlgorithmComposite::reset();
  _pitchContours->reset();
  _pitchContoursMelody->reset();
  // TODO shouldn't PoolStorage have a reset() method to clean the pool instead
  // of removing it manually? Furthermore, we do not need to call it in our
  // case, as PoolStorage is the part of the network.
  _pool.remove("internal.saliencebins");  // TODO is pool reset automatically?
  _pool.remove("internal.saliencevalues");// TODO --> no need to remove it here?
}

} // namespace streaming
} // namespace essentia
