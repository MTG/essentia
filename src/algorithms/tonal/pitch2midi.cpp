#include "pitch2midi.h"

using namespace essentia;
using namespace standard;

const char* Pitch2Midi::name = "Pitch2Midi";
const char* Pitch2Midi::category = "Input/Output";
const char* Pitch2Midi::description = DOC("");

void Pitch2Midi::configure() {
  _tuningFreq = parameter("tuningFreq").toReal();
  _transposition = parameter("transposition").toInt();
}

void Pitch2Midi::compute() {
  // get ref to input
  const Real& pitch = _pitch.get();
  const Real& loudness = _loudness.get();
  // get refs to outputs
  int& midiNoteNumber = _midiNoteNumber.get();
  int& midiNoteNumberTransposed = _midiNoteNumberTransposed.get();
  std::string& closestNoteName = _closestNoteName.get();
  std::string& closestNoteNameTransposed = _closestNoteNameTransposed.get();
  Real& closestPitch = _closestPitch.get();
  Real& diff = _diff.get();
  Real& cents = _cents.get();
  int& velocity = _velocity.get();

  Real _detectedPitch = pitch;
  Real _detectedLoudness = loudness;
    
  if (pitch <= 0) { _detectedPitch = 1e-05; }

  int idx = getMIDINoteIndex(_detectedPitch);
  int transposed_idx = idx + _transposition;
  midiNoteNumber = getMidiNoteNumberFromNoteIndex(idx);
  midiNoteNumberTransposed = getMidiNoteNumberFromNoteIndex(transposed_idx);
  closestNoteName = getClosestNoteName(idx);
  closestNoteNameTransposed = getClosestNoteName(transposed_idx);
  closestPitch = getClosestPitch(idx);
  diff = getDiff(closestPitch, _detectedPitch);
  cents = getCents(closestPitch, _detectedPitch);

  velocity = decibelsToVelocity(gainToDecibels(_detectedLoudness));
}

