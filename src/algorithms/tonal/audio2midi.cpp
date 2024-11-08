#include "audio2midi.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char *Audio2Midi::name = "Audio2Midi";
const char *Audio2Midi::category = "Pitch";
const char *Audio2Midi::description = DOC("Wrapper around Audio2Pitch and Pitch2Midi for real time application. This algorithm has a state that is used to estimate note on/off events based on consequent compute() calls.");

void Audio2Midi::configure()
{
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();
  _tuningFrequency = parameter("tuningFrequency").toInt();
  _pitchConfidenceThreshold = parameter("pitchConfidenceThreshold").toReal();
  _loudnessThreshold = parameter("loudnessThreshold").toReal();
  _transposition = parameter("transpositionAmount").toInt();
  _minOccurrenceRate = parameter("minOccurrenceRate").toReal();
  _midiBufferDuration = parameter("midiBufferDuration").toReal();
  _minNoteChangePeriod = parameter("minNoteChangePeriod").toReal();
  _minOnsetCheckPeriod = parameter("minOnsetCheckPeriod").toReal();
  _minOffsetCheckPeriod = parameter("minOffsetCheckPeriod").toReal();
  
  // define frameSize depending on sampleRate
  if (static_cast<int>(_sampleRate) <= 16000){
    _frameSize = 2048;
  }
  else if (static_cast<int>(_sampleRate) <= 24000){
    _frameSize = 4096;
  }
  else {
    _frameSize = 8192;
  }

  _applyTimeCompensation = parameter("applyTimeCompensation").toBool();

  _lowpass->configure(INHERIT("sampleRate"),
                      "cutoffFrequency", 1000);
  _framebuffer->configure("bufferSize", _frameSize);
  _audio2pitch->configure(INHERIT("sampleRate"),
                          "frameSize", _frameSize,
                          "pitchAlgorithm", _pitchAlgorithm,
                          "minFrequency", _minFrequency,
                          "maxFrequency", _maxFrequency,
                          INHERIT("pitchConfidenceThreshold"),
                          INHERIT("loudnessThreshold"));
  
  _pitch2midi->configure(INHERIT("sampleRate"),
                       INHERIT("hopSize"),
                       INHERIT("minOccurrenceRate"),
                       INHERIT("applyTimeCompensation"),
                       "minOnsetCheckPeriod", _minOnsetCheckPeriod,
                       "minOffsetCheckPeriod", _minOffsetCheckPeriod,
                       "minNoteChangePeriod", _minNoteChangePeriod,
                       "midiBufferDuration", _midiBufferDuration,
                       "minFrequency", _minFrequency,
                       "tuningFrequency", _tuningFrequency,
                       "transpositionAmount", _transposition);
}

void Audio2Midi::compute()
{
  // get ref to input
  const std::vector<Real> &frame = _frame.get();
  Real& pitch = _pitch.get();
  Real& loudness = _loudness.get();
  vector<string>& messageType = _messageType.get();
  vector<Real>& midiNoteNumber = _midiNoteNumber.get();
  vector<Real>& timeCompensation = _timeCompensation.get();

  _lowpass->input("signal").set(frame);
  _lowpass->output("signal").set(lpFrame);

  _framebuffer->input("frame").set(lpFrame);
  _framebuffer->output("frame").set(analysisFrame);

  _audio2pitch->input("frame").set(analysisFrame);
  _audio2pitch->output("pitch").set(pitch);
  _audio2pitch->output("pitchConfidence").set(pitchConfidence);
  _audio2pitch->output("loudness").set(loudness);
  _audio2pitch->output("voiced").set(voiced);

  _pitch2midi->input("pitch").set(pitch);
  _pitch2midi->input("voiced").set(voiced);
  _pitch2midi->output("midiNoteNumber").set(midiNoteNumber);
  _pitch2midi->output("timeCompensation").set(timeCompensation);
  _pitch2midi->output("messageType").set(messageType);
    
  _lowpass->compute();
  _framebuffer->compute();
  _audio2pitch->compute();
  _pitch2midi->compute();

}
