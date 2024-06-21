#ifndef ESSENTIA_PITCH2MIDI_H
#define ESSENTIA_PITCH2MIDI_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

  class Note {
    public:
      const Real midiNote;
    
      Note (Real midiNote) : midiNote(midiNote) {}
      ~Note () {}
  };

  class Pitch2Midi : public Algorithm {
    protected:
      // Inputs
      Input<Real> _pitch;
      Input<int> _voiced;

      // Outputs
      /*
      Output<int> _midiNoteNumberOut;
      Output<int> _previousMidiNoteNumberOut;
      Output<Real> _onsetTimeCompensation;
      Output<Real> _offsetTimeCompensation;
      Output<int> _messageType;
      */
      
      // define outputs as vectors - index vector connects the ouputs
      // output vector for _messageType ("note_off", "note_on")
      Output<std::vector<std::string> > _messageTypeOut;
      // output vector for midiNoteNumber (<note_off>, <note_on>)
      Output<std::vector<Real> > _midiNoteNumberOut;
      // output vector for timeCompensation (offsetCompensation, onsetCompensation)
      Output<std::vector<Real> > _timeCompensationOut;

      bool _noteOn;
      bool _noteOff;
      
      Algorithm* _framebuffer;

      // parameters
      Real _sampleRate;
      int _hopSize;
      Real _minFrequency;
      Real _minOcurrenceRate;
      Real _minOnsetCheckPeriod;
      Real _minOffsetCheckPeriod;
      Real _minNoteChangePeriod;
      Real _bufferDuration = 0.015;// 15ms default
      bool _applyCompensation = true;
      // former Pitch2Midi params
      Real _tuningFreq;
      int _transposition;
      
      // other
      int _capacity;
      int _minCapacity = 3;
      bool _NOTED_ON = false;
      std::vector<Real> _maxVoted;
      bool _COHERENCE;
      Note* note = new Note(0);
      Note* dnote_ = new Note(0);
      Real _detectedPitch;

      // Containers
      std::vector<Real> _midiNoteNumberVector; // always size 1, but frameBuffer algo expects vectors as input
      std::vector<Real> _buffer;
      std::vector<Real> _midiNoteNumberBin;     // should be a vector of integers
      std::vector<Real> _timeCompensationBin;
      std::vector<std::string> _messageTypeBin;

      int capacity();
      bool hasCoherence();
      void getMaxVoted();
      
      void updateDnote();
      // condition checkers
      bool isMaxVotedZero();
      bool isCurrentMidiNoteEqualToMaxVoted();
      bool isMaxVotedCountGreaterThanMinOcurrenceRate();
      void setOutputs(int midiNoteNumber, float onsetTimeCompensation, float offsetTimeCompensation);
      
      Real _minOnsetCheckThreshold;
      Real _minOffsetCheckThreshold;
      Real _minNoteChangeThreshold;
      
      int _unvoicedFrameCounter;
      int _offsetCheckCounter;
      int _onsetCheckCounter;
      
      Real _frameTime;
      Real _minOcurrenceRateThreshold;
      Real _minOcurrenceRatePeriod;

      // former Pitch2Midi outputs, now interal vars
      int _midiNoteNumberTransposed;

      // TODO: replace by essentiamath conversions
      int inline getMIDINoteIndex(Real& pitch) {
        return (int) round(log2(pitch / _tuningFreq) * 12); // it should be added +69 to get midiNote
      }

      // convert pitch in MIDI note
      int inline getMidiNoteNumberFromNoteIndex(int idx) {
        return 69 + idx;
      }

    public:
      Pitch2Midi() : _maxVoted(2), _midiNoteNumberVector(1) {
        declareInput(_pitch, "pitch", "pitch given in Hz for conversion");
        declareInput(_voiced, "voiced", "whether the frame is voiced or not");
        /*declareOutput(_midiNoteNumberOut, "midiNoteNumber", "detected MIDI note number, as integer, in range [0,127]");
        declareOutput(_previousMidiNoteNumberOut, "previousMidiNoteNumber", "detected MIDI note number in previous compute call, as integer, in range [0,127]");
        declareOutput(_onsetTimeCompensation, "onsetTimeCompensation", "time to be compensated in the onset message");
        declareOutput(_offsetTimeCompensation, "offsetTimeCompensation", "time to be compensated in the offset message");
        declareOutput(_messageType, "messageType", "defines MIDI message type, as integer, where 0: offset, 1: onset, 2: offset-onset");*/
        declareOutput(_messageTypeOut, "messageType", "the output of MIDI message type, as string, {noteoff, noteon, noteoff-noteon}");
        declareOutput(_midiNoteNumberOut, "midiNoteNumber", "the output of detected MIDI note number, as integer, in range [0,127]");
        declareOutput(_timeCompensationOut, "timeCompensation", "time to be compensated in the messages");
      }

      // TODO: redefine outputs: messageType, timeCompensations, midiNoteNumbers
      
      ~Pitch2Midi() {
        delete _framebuffer;

        delete note;
        delete dnote_;
      };

      void declareParameters() {
        declareParameter("sampleRate", "sample rate of incoming audio frames", "[8000,inf)", 44100);
        declareParameter("hopSize", "analysis hop size in samples, equivalent to I/O buffer size", "[1,inf)", 128);
        declareParameter("minFrequency", "minimum detectable frequency", "[20,20000]", 60.f);
        declareParameter("minOcurrenceRate", "minimum number of times a midi note has to ocur compared to total capacity", "[0,1]", 0.5f);
        declareParameter("midiBufferDuration", "duration in seconds of buffer used for voting in the note toggle detection algorithm", "[0.005,0.5]", 0.015); // 15ms
        declareParameter("minNoteChangePeriod", "minimum time to wait until a note change is detected", "(0,1]", 0.030);
        declareParameter("minOnsetCheckPeriod", "minimum time to wait until an onset is detected", "(0,1]", 0.075);
        declareParameter("minOffsetCheckPeriod", "minimum time to wait until an offset is detected", "(0,1]", 0.2);
        declareParameter("applyTimeCompensation", "whether to apply time compensation in the timestamp of the note toggle messages.", "{true,false}", true);
        // former Pitch2Midi params
        declareParameter("tuningFreq", "reference tuning frequency in Hz", "{432,440}", 440);
        declareParameter("transpositionAmount", "amount of semitones to apply for transposed instruments", "(-69,50)", 0);
      }

      void configure();
      void compute();
      void getMidiNoteNumber(Real pitch);

      void push(int midiNoteNumber);

      const Note* dnote() const { return dnote_; }

      static const char* name;
      static const char* category;
      static const char* description;
  };


} // namespace standard
} // namespace essentia

#endif
