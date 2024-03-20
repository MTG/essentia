#ifndef ESSENTIA_PITCH2MIDI_H
#define ESSENTIA_PITCH2MIDI_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Pitch2Midi : public Algorithm {

  protected:
    Input<Real> _pitch;
    Input<Real> _loudness;
    Output<int> _midiNoteNumber;
    Output<int> _midiNoteNumberTransposed;
    Output<std::string> _closestNoteName;
    Output<std::string> _closestNoteNameTransposed;
    Output<Real> _closestPitch;
    Output<Real> _diff;
    Output<Real> _cents;
    Output<int> _velocity;

    Real _tuningFreq;
    int _transposition;

    const std::vector<std::string> ALL_NOTES { "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#" };
    Real hearing_threshold {-96.0f};  // we consider the 16-bits dynamic range - 96dB(SPL)

    int inline getMIDINoteIndex(Real& pitch) {
      return (int) round(log2(pitch / _tuningFreq) * 12); // it should be added +69 to get midiNote
    }

    // convert pitch in MIDI note
    int inline getMidiNoteNumberFromNoteIndex(int idx) {
      return 69 + idx;
    }

    std::string inline getClosestNoteName(int i) {
      int idx = abs(i) % 12;
      int octave = 4 + floor((i + 9) / 12.f);
      if (i < 0)
          idx = abs(idx - 12) % 12;   // keep the index in music notes array when i is negative
      std::string closest_note = ALL_NOTES[idx] + std::to_string(octave);
      
      // TODO: for the line above check this https://forum.juce.com/t/efficiency-of-string-concatenation-vs-getting-a-substring/18296
      return closest_note;
    }

    Real inline getClosestPitch(int i) {
      return _tuningFreq * powf(2, i / 12.f);
    }

    Real inline getDiff(Real& closest, Real& detected) {
      return round(detected - closest);
    }

    // convert pitch in cents
    Real inline getCents(Real& frequency_a, Real& frequency_b) {
      return 1200 * log2(frequency_b / frequency_a);
    }

    // convert loudness [dB] in to velocity
    int inline decibelsToVelocity (Real decibels) {
        int velocity = 0;
        if (decibels > hearing_threshold)
            velocity = (int)((hearing_threshold - decibels) * 127 / hearing_threshold);  // decibels should be negative
        return velocity;
    }
    
    // convert gain to decibels
    Real inline gainToDecibels(Real& gain){
        return 20 * log10(gain);;
    }

  public:
    Pitch2Midi() {
      declareInput(_pitch, "pitch", "pitch given in Hz for conversion");
      declareInput(_loudness, "loudness", "loudness given in dB for velocity conversion");
      declareOutput(_midiNoteNumber, "midiNoteNumber", "midi note number, as integer, in range [0,127]");
      declareOutput(_midiNoteNumberTransposed, "midiNoteNumberTransposed", "midi note number with applied transposition, as integer, in range [0,127]");
      declareOutput(_closestNoteName, "closestNoteName", "pitch class and octave number to detected pitch, as string (e.g. A4)");
      declareOutput(_closestNoteNameTransposed, "closestNoteNameTransposed", "pitch class and octave number to detected pitch, with applied transposition, as string (e.g. A4)");
      declareOutput(_closestPitch, "closestPitch", "equal-tempered pitch closest to detected pitch, in Hz");
      declareOutput(_diff, "diff", "difference between pitch and closestPitch, in Hz");
      declareOutput(_cents, "cents", "difference between pitch and closestPitch, in cents (1/100 of a semitone)");
      declareOutput(_velocity, "velocity", "control message over the feel and volume of MIDI notes, as integer, in range [0,127])");
      
    }
    ~Pitch2Midi() {
    }

    void declareParameters() {
      declareParameter("tuningFreq", "reference tuning frequency in Hz", "{432,440}", 440);
      declareParameter("transposition", "amount of semitones to apply for transposed instruments", "(-69,50)", 0);
    }

    void configure();
    void compute();

    static const char* name;
    static const char* category;
    static const char* description;
};

} // namespace standard
} // namespace essentia

#endif
