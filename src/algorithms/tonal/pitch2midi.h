#ifndef ESSENTIA_PITCH2MIDI_H
#define ESSENTIA_PITCH2MIDI_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

  class Pitch2Midi : public Algorithm {
    protected:
      // Inputs
      Input<Real> _pitch;
      Input<int> _voiced;

      // Outputs
      Output<std::vector<std::string> > _messageType;
      Output<std::vector<Real> > _midiNoteNumber;
      Output<std::vector<Real> > _timeCompensation;

      bool _noteOn;
      bool _noteOff;
      
      Algorithm* _framebuffer;

      // parameters
      Real _sampleRate;
      int _hopSize;
      Real _minFrequency;
      Real _minOccurrenceRate;
      Real _minOnsetCheckPeriod;
      Real _minOffsetCheckPeriod;
      Real _minNoteChangePeriod;
      Real _bufferDuration;
      bool _applyCompensation;
      // former Pitch2Midi params
      Real _tuningFreq;
      int _transposition;
      
      // other
      int _capacity;
      int _minCapacity = 3;
      bool _NOTED_ON = false;
      std::vector<Real> _maxVoted;
      bool _COHERENCE;
      Real note = 0.0;
      Real dnote = 0.0;
      Real _detectedPitch;

      // Containers
      std::vector<Real> _midiNoteNumberVector; // always size 1, but frameBuffer algo expects vectors as input
      std::vector<Real> _buffer;

      int capacity();
      bool hasCoherence();
      void getMaxVoted();
      
      void updateDnote();
      void setOutputs(Real midiNoteNumber, float onsetTimeCompensation, float offsetTimeCompensation);
      
      Real _minOnsetCheckThreshold;
      Real _minOffsetCheckThreshold;
      Real _minNoteChangeThreshold;
      
      int _unvoicedFrameCounter;
      int _offsetCheckCounter;
      int _onsetCheckCounter;
      
      Real _frameTime;
      Real _minOccurrenceRateThreshold;
      Real _minOccurrenceRatePeriod;

      // former Pitch2Midi outputs, now interal vars
      Real _midiNoteNumberTransposed;

    public:
      Pitch2Midi() : _maxVoted(2), _midiNoteNumberVector(1) {
        declareInput(_pitch, "pitch", "pitch given in Hz for conversion");
        declareInput(_voiced, "voiced", "whether the frame is voiced or not, (0, 1)");
        declareOutput(_messageType, "messageType", "the output of MIDI message type, as string, {noteoff, noteon, noteoff-noteon}");
        declareOutput(_midiNoteNumber, "midiNoteNumber", "the output of detected MIDI note number, as integer, in range [0,127]");
        declareOutput(_timeCompensation, "timeCompensation", "time to be compensated in the messages");
      }
      
      ~Pitch2Midi() {
        delete _framebuffer;
      };

      void declareParameters() {
        declareParameter("sampleRate", "Audio sample rate", "[8000,inf)", 44100);
        declareParameter("hopSize", "Pitch Detection analysis hop size in samples, equivalent to I/O buffer size", "[1,inf)", 128);
        declareParameter("minFrequency", "minimum detectable frequency", "[20,20000]", 60.0);
        declareParameter("minOccurrenceRate", "minimum number of times a midi note has to ocur compared to total capacity", "[0,1]", 0.5);
        declareParameter("midiBufferDuration", "duration in seconds of buffer used for voting in the note toggle detection algorithm", "[0.005,0.5]", 0.015); // 15ms
        declareParameter("minNoteChangePeriod", "minimum time to wait until a note change is detected (s)", "(0,1]", 0.030);
        declareParameter("minOnsetCheckPeriod", "minimum time to wait until an onset is detected (s)", "(0,1]", 0.075);
        declareParameter("minOffsetCheckPeriod", "minimum time to wait until an offset is detected (s)", "(0,1]", 0.2);
        declareParameter("applyTimeCompensation", "whether to apply time compensation in the timestamp of the note toggle messages.", "{true,false}", true);
        // former Pitch2Midi params
        declareParameter("tuningFrequency", "reference tuning frequency in Hz", "{432,440}", 440);
        declareParameter("transpositionAmount", "Apply transposition (in semitones) to the detected MIDI notes.", "(-69,50)", 0);
      }

      void configure();
      void compute();
      void inline getMidiNoteNumber(Real pitch);

      void push(Real midiNoteNumber);

      static const char* name;
      static const char* category;
      static const char* description;
  };


} // namespace standard
} // namespace essentia

#endif
