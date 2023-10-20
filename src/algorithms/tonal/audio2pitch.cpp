#include "audio2pitch.h"

using namespace essentia;
using namespace standard;

const char* Audio2Pitch::name = "Audio2Pitch";
const char* Audio2Pitch::category = "Pitch";
const char* Audio2Pitch::description = DOC("Extractor algorithm to compute pitch with several possible pitch algorithms, specifically targeted for real-time pitch detection on saxophone signals.");

void Audio2Pitch::configure() {

  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();
  _pitchAlgorithmName = parameter("pitchAlgorithm").toString();
  _loudnessAlgorithmName = parameter("loudnessAlgorithm").toString();

  if (_maxFrequency > _sampleRate*0.5 ) {
    throw EssentiaException("Audio2Pitch: Max frequency cannot be higher than Nyquist frequency");
  }
  if (_maxFrequency <= _minFrequency) {
    throw EssentiaException("Audio2Pitch: Max frequency cannot be lower than min frequency");
  }

  if (_pitchAlgorithmName != "pyin_fft" && _pitchAlgorithmName != "pyin") {
    E_INFO("Audio2Pitch: 'pitchAlgorithm' = "<<_pitchAlgorithmName<<"\n");
    throw EssentiaException("Audio2Pitch: Bad 'pitchAlgorithm' parameter");
  }
  if (_pitchAlgorithmName == "pyin_fft") _isSpectral = true;
  if (_pitchAlgorithmName == "pyin") _isSpectral = false;

  if (_isSpectral) {
    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _pitchAlgorithm = AlgorithmFactory::create("PitchYinFFT");

    _windowing->configure("type", "hann",
                          "size", _frameSize);
    _spectrum->configure("size", _frameSize);
  }
  else {
    _pitchAlgorithm = AlgorithmFactory::create("PitchYin");
  }

  if (_loudnessAlgorithmName == "loudness") {
    _loudnessAlgorithm = AlgorithmFactory::create("Loudness");
  }
  else if (_loudnessAlgorithmName == "rms") {
    _loudnessAlgorithm = AlgorithmFactory::create("RMS");
  }
  else {
    E_INFO("Audio2Pitch: 'loudnessAlgorithm' = "<<_loudnessAlgorithmName<<"\n");
    throw EssentiaException("Audio2Pitch: Bad 'loudnessAlgorithm' parameter");
  }

  _pitchAlgorithm->configure(INHERIT("frameSize"),
                             INHERIT("maxFrequency"),
                             INHERIT("minFrequency"),
                             INHERIT("sampleRate"));
}

void Audio2Pitch::compute() {
  const std::vector<Real>& frame = _frame.get();
  Real& pitch = _pitch.get();
  Real& pitchConfidence = _pitchConfidence.get();
  Real& loudness = _loudness.get();

  if (frame.empty()) {
    throw EssentiaException("Audio2Pitch: cannot compute the pitch of an empty frame");
  }

  if (frame.size() == 1) {
    throw EssentiaException("Audio2Pitch: cannot compute the pitch of a frame of size 1");
  }

  if (_loudnessAlgorithmName == "loudness") {
    _loudnessAlgorithm->input("signal").set(frame);
    _loudnessAlgorithm->output("loudness").set(loudness);
  }
  else {
    _loudnessAlgorithm->input("array").set(frame);
    _loudnessAlgorithm->output("rms").set(loudness);
  }
  _loudnessAlgorithm->compute();

  std::vector<Real> windowedFrame, spectrum;
  if (_isSpectral) {
    _windowing->input("frame").set(frame);
    _windowing->output("frame").set(windowedFrame);
    _windowing->compute();
    _spectrum->input("frame").set(windowedFrame);
    _spectrum->output("spectrum").set(spectrum);
    _spectrum->compute();
    _pitchAlgorithm->input("spectrum").set(spectrum);
  }
  else {
    _pitchAlgorithm->input("signal").set(frame);
  }

  _pitchAlgorithm->output("pitch").set(pitch);
  _pitchAlgorithm->output("pitchConfidence").set(pitchConfidence);
  _pitchAlgorithm->compute();

}
