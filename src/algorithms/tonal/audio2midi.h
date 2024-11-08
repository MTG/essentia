#ifndef ESSENTIA_AUDIO2MIDI_H
#define ESSENTIA_AUDIO2MIDI_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

  class Audio2Midi : public Algorithm {
    protected:
      Input<std::vector<Real>> _frame;
      Output<Real> _pitch;
      Output<Real> _loudness;
      Output<std::vector<std::string> > _messageType;
      Output<std::vector<Real> > _midiNoteNumber;
      Output<std::vector<Real> > _timeCompensation;

      Algorithm* _lowpass;
      Algorithm* _framebuffer;
      Algorithm* _audio2pitch;
      Algorithm* _pitch2midi;

      Real _sampleRate;
      int _frameSize;
      int _hopSize;
      std::string _pitchAlgorithm = "pitchyinfft";
      std::string _loudnessAlgorithm = "rms";
      Real _minFrequency;
      Real _maxFrequency;
      int _tuningFrequency;
      Real _pitchConfidenceThreshold, _loudnessThreshold, _minOccurrenceRate;
      Real _midiBufferDuration;
      Real _minNoteChangePeriod;
      Real _minOnsetCheckPeriod;
      Real _minOffsetCheckPeriod;

      bool _applyTimeCompensation;
      int _transposition;

      // Containers
      std::vector<Real> lpFrame, analysisFrame;
      Real pitch, pitchConfidence, loudness;
      std::vector<Real> midiNoteNumber, timeCompensation;
      std::vector<std::string> messageType;
      Real onsetTimeCompensation, offsetTimeCompensation;
      
      int voiced;
      
    public:
      Audio2Midi() {
        declareInput(_frame, "frame", "the input frame to analyse");
        declareOutput(_pitch, "pitch", "pitch given in Hz");
        declareOutput(_loudness, "loudness", "detected loudness in decibels");
        declareOutput(_messageType, "messageType", "the output of MIDI message type, as string, {noteoff, noteon, noteoff-noteon}");
        declareOutput(_midiNoteNumber, "midiNoteNumber", "the output of detected MIDI note number, as integer, in range [0,127]");
        declareOutput(_timeCompensation, "timeCompensation", "time to be compensated in the messages");
          
        _lowpass = AlgorithmFactory::create("LowPass");
        _framebuffer = AlgorithmFactory::create("FrameBuffer");
        _audio2pitch = AlgorithmFactory::create("Audio2Pitch");
        _pitch2midi = AlgorithmFactory::create("Pitch2Midi");
      }

      ~Audio2Midi() {
        delete _lowpass;
        delete _framebuffer;
        delete _audio2pitch;
        delete _pitch2midi;
      }

      void declareParameters() {
        declareParameter("sampleRate", "sample rate of incoming audio frames", "[8000,inf)", 44100);
        declareParameter("hopSize", "equivalent to I/O buffer size", "[1,inf)", 32);
        declareParameter("minFrequency", "minimum frequency to detect in Hz", "[10,20000]", 60.0);
        declareParameter("maxFrequency", "maximum frequency to detect in Hz", "[10,20000]", 2300.0);
        declareParameter("tuningFrequency", "tuning frequency for semitone index calculation, corresponding to A3 [Hz]", "{432,440}", 440);
        declareParameter("pitchConfidenceThreshold", "level of pitch confidence above which note ON/OFF start to be considered", "[0,1]", 0.25);
        declareParameter("loudnessThreshold", "loudness level above/below which note ON/OFF start to be considered, in decibels", "[-inf,0]", -51.0);
        declareParameter("transpositionAmount", "Apply transposition (in semitones) to the detected MIDI notes.", "(-69,50)", 0);
        declareParameter("minOccurrenceRate", "rate of predominant pitch occurrence in MidiPool buffer to consider note ON event", "[0,1]", 0.5);
        declareParameter("midiBufferDuration", "duration in seconds of buffer used for voting in MidiPool algorithm", "[0.005,0.5]", 0.05); // 15ms
        declareParameter("minNoteChangePeriod", "minimum time to wait until a note change is detected (testing only)", "(0,1]", 0.030);
        declareParameter("minOnsetCheckPeriod", "minimum time to wait until an onset is detected (testing only)", "(0,1]", 0.075);
        declareParameter("minOffsetCheckPeriod", "minimum time to wait until an offset is detected (testing only)", "(0,1]", 0.2);
        declareParameter("applyTimeCompensation", "whether to apply time compensation correction to MIDI note detection", "{true,false}", true);
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
