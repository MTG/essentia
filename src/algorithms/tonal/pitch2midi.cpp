#include "pitch2midi.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Pitch2Midi::name = "Pitch2Midi";
const char* Pitch2Midi::category = "Pitch";
const char *Pitch2Midi::description = DOC("This algorithm estimates the midi note ON/OFF detection from raw pitch and voiced values, using midi buffer and uncertainty checkers.");

void Pitch2Midi::configure()
{
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
  _minFrequency = parameter("minFrequency").toReal();
  _minOcurrenceRate = parameter("minOcurrenceRate").toReal();
  _bufferDuration = parameter("midiBufferDuration").toReal();
  _minOnsetCheckPeriod = parameter("minOnsetCheckPeriod").toReal();
  _minOffsetCheckPeriod = parameter("minOffsetCheckPeriod").toReal();
  _minNoteChangePeriod = parameter("minNoteChangePeriod").toReal();
  _applyCompensation = parameter("applyTimeCompensation").toBool();
  // former Pitch2Midi only parameters
  _tuningFreq = parameter("tuningFreq").toReal();
  _transposition = parameter("transpositionAmount").toInt();

  _frameTime = _hopSize / _sampleRate;
  _minOnsetCheckThreshold = _minOnsetCheckPeriod / _frameTime;
  _minOffsetCheckThreshold = _minOffsetCheckPeriod / _frameTime;
  _minNoteChangeThreshold = _minNoteChangePeriod / _frameTime;
    
  E_INFO("_minOnsetCheckThreshold: " << _minOnsetCheckThreshold);
  E_INFO("_minOffsetCheckThreshold: " << _minOffsetCheckThreshold);
  E_INFO("_minNoteChangeThreshold: " << _minNoteChangeThreshold);

  _unvoicedFrameCounter = 0;
  _offsetCheckCounter = 0;
  _onsetCheckCounter = 0;
    
  _minOcurrenceRatePeriod = _minOcurrenceRate * _bufferDuration;
  _minOcurrenceRateThreshold = _minOcurrenceRatePeriod / _frameTime;

  _capacity = capacity();
  _framebuffer = AlgorithmFactory::create("FrameBuffer");
  _framebuffer->configure("bufferSize", _capacity);

}

void Pitch2Midi::getMidiNoteNumber(Real pitch)
{
  _detectedPitch = pitch;
    
  if (pitch <= 0) { _detectedPitch = 1e-05; }

  /*
   int idx = getMIDINoteIndex(_detectedPitch);
  int transposed_idx = idx + _transposition;
  _midiNoteNumberTransposed = getMidiNoteNumberFromNoteIndex(transposed_idx);
   */
  int idx = hz2midi(pitch, _tuningFreq);
  _midiNoteNumberTransposed = idx + _transposition;
}

int Pitch2Midi::capacity()
{
  float hopSizeFloat = _hopSize; // ensure no int/int division happens
  float sampleRateFloat = _sampleRate;
  int c = static_cast<int>( round( sampleRateFloat / hopSizeFloat * _bufferDuration ) );
  return max(_minCapacity, c);
}

// this should NOT be called until framebuffer.compute has been called
bool Pitch2Midi::hasCoherence()
{
  Real sum = accumulate(_buffer.begin(), _buffer.end(), 0.0);
  if (sum / _capacity == _buffer[0]) {
      return true;
  }
  return false;
}

// this should NOT be called until framebuffer.compute has been called and _capacity has been set on the configure
void Pitch2Midi::getMaxVoted()
{
  map<Real, Real> counts;
  for (Real value : _buffer) {
    counts[value]++;
  }

  Real maxCount = 0;
  Real maxValue = 0;
  for (auto& pair : counts) {
    if (pair.second > maxCount) {
      maxCount = pair.second;
      maxValue = pair.first;
    }
  }

  _maxVoted[0] = maxValue;
  _maxVoted[1] = maxCount / _capacity;
}

bool Pitch2Midi::isMaxVotedZero() {
  return _maxVoted[0] == 0.0;
}

bool Pitch2Midi::isCurrentMidiNoteEqualToMaxVoted() {
  return note->midiNote == _maxVoted[0];
}

bool Pitch2Midi::isMaxVotedCountGreaterThanMinOcurrenceRate() {
  return _maxVoted[1] > _minOcurrenceRate;
}

void Pitch2Midi::setOutputs(int midiNoteNumber, float onsetTimeCompensation, float offsetTimeCompensation) {
  // get refs to outputs
  int& midiNoteNumberOut = _midiNoteNumberOut.get();
  int& previousMidiNoteNumberOut = _previousMidiNoteNumberOut.get();
  Real& onsetTimeCompensationOut = _onsetTimeCompensation.get();
  Real& offsetTimeCompensationOut = _offsetTimeCompensation.get();
  int& messageTypeOut = _messageType.get();

  if (midiNoteNumber <= 0 && midiNoteNumber >= 127) {
    E_INFO("SCAPE");
    return;
  }

  if (!_applyCompensation) {
    onsetTimeCompensation = 0.f;
    offsetTimeCompensation = 0.f;
  }

  midiNoteNumberOut = midiNoteNumber;
  previousMidiNoteNumberOut = dnote_->midiNote;
  onsetTimeCompensationOut = onsetTimeCompensation;
  offsetTimeCompensationOut = offsetTimeCompensation;

  if (_noteOff && !_noteOn) {
    messageTypeOut = 0;
  }

  if (_noteOn && !_noteOff) {
    messageTypeOut = 1;
  }

  if (_noteOn && _noteOff) {
    messageTypeOut = 2;
  }
}

void Pitch2Midi::push(int midiNoteNumber) {
    // push new MIDI note number in the MIDI buffer
    _midiNoteNumberVector[0] = static_cast<Real>(midiNoteNumber);
    _framebuffer->input("frame").set(_midiNoteNumberVector);
    _framebuffer->output("frame").set(_buffer);
    _framebuffer->compute();
}

void Pitch2Midi::compute()
{
  // former MidiPool inputs are now Pitch2Midi internal vars
  // all we need is to run the conversions:
  const Real& pitch = _pitch.get();
  const int& voiced = _voiced.get();
  getMidiNoteNumber(pitch);

  int& messageTypeOut = _messageType.get();
  // initialize messageType to null value
  messageTypeOut = -1;
    
  // refresh note_on and note_off timestamps
  _noteOn = false;
  _noteOff = false;
  
  // unvoiced frame detection
  if (!voiced) {
    if ( _NOTED_ON ) {
      _unvoicedFrameCounter++;
      if (_unvoicedFrameCounter > _minNoteChangeThreshold) {
        _NOTED_ON = false;
        _noteOff = true;
        updateDnote();
        setOutputs(dnote_->midiNote, 0.f, _minNoteChangePeriod);
        E_INFO("offset(unvoiced frame)");
        _unvoicedFrameCounter = 0;
        _offsetCheckCounter = 0;
        _onsetCheckCounter = 0;
      }
    } else {
      _unvoicedFrameCounter = 0;
      push(0);    // push 0th MIDI note to remove the past
      _offsetCheckCounter = 0;
      _onsetCheckCounter = 0;
    }
    return;
  }

  _unvoicedFrameCounter = 0;
    
  // push new MIDI note number in the MIDI buffer
  push(_midiNoteNumberTransposed);

  // update max_voting
  getMaxVoted();


  E_INFO("onset thresholds: "<< _onsetCheckCounter <<" - " << _minOnsetCheckThreshold);
  E_INFO("offset thresholds: "<< _offsetCheckCounter <<" - " << _minOffsetCheckThreshold);
  // analyze pitch buffer
  if (hasCoherence() && _NOTED_ON) {
    if (isCurrentMidiNoteEqualToMaxVoted()) {
      _offsetCheckCounter = 0;
      _onsetCheckCounter = 0;
    }
    else {
      // IMPORTANT: this hardly happens so if hasCoherence() current MIDI note is equals to max voted.
      _offsetCheckCounter++;
      if (_offsetCheckCounter > _minOffsetCheckThreshold) {
        updateDnote();
        delete note;
        note = new Note(_buffer[0]);
        _noteOff = true;
        _noteOn = true;
        _offsetCheckCounter = 0;
        _onsetCheckCounter = 0;
        E_INFO("off-onset(coherent & NOTED)");
      }
    }
    setOutputs(_midiNoteNumberTransposed, _minOffsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }

  if (hasCoherence() && !_NOTED_ON) {

    _onsetCheckCounter++;
      
    if (_onsetCheckCounter > _minOnsetCheckThreshold){
      delete note;
      note = new Note(_buffer[0]);
      _noteOn = true;
      _NOTED_ON = true;
      E_INFO("onset(coherent & !NOTED): "<< _onsetCheckCounter <<" - " << _minOnsetCheckThreshold);
      _onsetCheckCounter = 0;
      _offsetCheckCounter = 0;
    }
    setOutputs(_midiNoteNumberTransposed, _minOnsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }

  if (!hasCoherence() && _NOTED_ON) {
    if (!isMaxVotedZero()) {
      _onsetCheckCounter++;
      // combines checker with minOcurrenceRate
      if (_onsetCheckCounter > _minOcurrenceRateThreshold){
        _noteOff = true;
        _noteOn = true;
        _NOTED_ON = true;
        updateDnote();
        delete note;
        note = new Note(_maxVoted[0]);
        E_INFO("off-onset(uncoherent & NOTED): " << _onsetCheckCounter << " - " << _minOcurrenceRateThreshold);
        _offsetCheckCounter = 0;
        _onsetCheckCounter = 0;
      }
    }
    setOutputs(_midiNoteNumberTransposed, _minOcurrenceRatePeriod, _minOcurrenceRatePeriod);
    return;
  }

  if (!hasCoherence() && !_NOTED_ON) {
    if (isMaxVotedCountGreaterThanMinOcurrenceRate()) {
      _onsetCheckCounter++;

      if (_onsetCheckCounter > _minOnsetCheckThreshold) {
        if (!isMaxVotedZero()) {
          delete note;
          note = new Note(_maxVoted[0]);
          _NOTED_ON = true;
          _noteOn = true;
          E_INFO("onset(uncoherent & unNOTED)");
          _onsetCheckCounter = 0;
          _offsetCheckCounter = 0;
        }
      }
    }
    setOutputs(_midiNoteNumberTransposed, _minOnsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }
  E_INFO("Compute() -END");
}




void Pitch2Midi::updateDnote () {
  delete dnote_;
  dnote_ = new Note(*note);
}
