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
  _minOccurrenceRate = parameter("minOccurrenceRate").toReal();
  _bufferDuration = parameter("midiBufferDuration").toReal();
  _minOnsetCheckPeriod = parameter("minOnsetCheckPeriod").toReal();
  _minOffsetCheckPeriod = parameter("minOffsetCheckPeriod").toReal();
  _minNoteChangePeriod = parameter("minNoteChangePeriod").toReal();
  _applyCompensation = parameter("applyTimeCompensation").toBool();
  // former Pitch2Midi only parameters
  _tuningFreq = parameter("tuningFrequency").toReal();
  _transposition = parameter("transpositionAmount").toInt();

  _frameTime = _hopSize / _sampleRate;
  _minOnsetCheckThreshold = _minOnsetCheckPeriod / _frameTime;
  _minOffsetCheckThreshold = _minOffsetCheckPeriod / _frameTime;
  _minNoteChangeThreshold = _minNoteChangePeriod / _frameTime;

  _unvoicedFrameCounter = 0;
  _offsetCheckCounter = 0;
  _onsetCheckCounter = 0;
    
  _minOccurrenceRatePeriod = _minOccurrenceRate * _bufferDuration;
  _minOccurrenceRateThreshold = _minOccurrenceRatePeriod / _frameTime;

  // estimate buffer capacity
  int c = static_cast<int>( round( _sampleRate / float(_hopSize) * _bufferDuration ) );
  _capacity =  max(_minCapacity, c);
  _framebuffer = AlgorithmFactory::create("FrameBuffer");
  _framebuffer->configure("bufferSize", _capacity);

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
  // estimates the max voted MIDI note in the midi note buffer
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

void Pitch2Midi::setOutputs(Real midiNoteNumberValue, float onsetTimeCompensation, float offsetTimeCompensation) {
  vector<string>& messageType = _messageType.get();
  vector<Real>& midiNoteNumber = _midiNoteNumber.get();
  vector<Real>& timeCompensation = _timeCompensation.get();
    
  // reuse bins
  messageType.resize(0);
  midiNoteNumber.resize(0);
  timeCompensation.resize(0);

  // TODO: this is not clear because it might remove an note_off message which is defined by dnote.
  //#! it would be better just to provide some code for midiNoteNumbre when this happens
  if (midiNoteNumberValue <= 0 && midiNoteNumberValue >= 127) {
    //E_INFO("SCAPE");
    return;
  }

  // let's define first the message type
  if (_noteOff) {
    messageType.push_back("note_off");
  }

  if (_noteOn) {
    messageType.push_back("note_on");
  }

  if (!_applyCompensation) {
    onsetTimeCompensation = 0.f;
    offsetTimeCompensation = 0.f;
  }

  midiNoteNumber.push_back(dnote);
  midiNoteNumber.push_back(midiNoteNumberValue);
  timeCompensation.push_back(offsetTimeCompensation);
  timeCompensation.push_back(onsetTimeCompensation);
}

void Pitch2Midi::push(Real midiNoteNumber) {
    // push new MIDI note number in the MIDI buffer
    _midiNoteNumberVector[0] = midiNoteNumber;
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
    
  // do sanity checks
  if (pitch < 0) {
    throw EssentiaException("Pitch2Midi: specified duration of the input signal must be non-negative");
  }

  _detectedPitch = pitch;
  if (pitch < 0) { _detectedPitch = 1e-05; }
  _midiNoteNumberTransposed = hz2midi(_detectedPitch, _tuningFreq) + _transposition;
    
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
        setOutputs(dnote, 0.0, _minNoteChangePeriod);
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

  // analyze pitch buffer
  if (hasCoherence() && _NOTED_ON) {
    if (note == _maxVoted[0]) {
      _offsetCheckCounter = 0;
      _onsetCheckCounter = 0;
    }
    else {
      // IMPORTANT: this hardly happens so if hasCoherence() current MIDI note is equals to max voted.
      _offsetCheckCounter++;
      if (_offsetCheckCounter > _minOffsetCheckThreshold) {
        _NOTED_ON = true;
        if (note != _buffer[0]){  // avoid note slicing effect
            updateDnote();
            note = _buffer[0];
            _noteOff = true;
            _noteOn = true;
        }
        _offsetCheckCounter = 0;
        _onsetCheckCounter = 0;
        //E_WARNING("off-onset(" << _buffer[0] << ", coherent & NOTED)");
      }
    }
    // in coherence output the _midiNoteNumberTransposed coincides with _buffer[0] value
    setOutputs(_midiNoteNumberTransposed, _minOffsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }

  if (hasCoherence() && !_NOTED_ON) {

    _onsetCheckCounter++;
      
    if (_onsetCheckCounter > _minOnsetCheckThreshold){
      note = _buffer[0];
      _noteOn = true;
      _NOTED_ON = true;
      //E_INFO("onset(" << _buffer[0] << ", coherent & !NOTED): "<< _onsetCheckCounter <<" - " << _minOnsetCheckThreshold);
      _onsetCheckCounter = 0;
      _offsetCheckCounter = 0;
    }
    // in coherence output the _midiNoteNumberTransposed coincides with _buffer[0] value
    setOutputs(_midiNoteNumberTransposed, _minOnsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }

  if (!hasCoherence() && _NOTED_ON) {
    if (_maxVoted[0] != 0.0) {
      _onsetCheckCounter++;
      // combines checker with minOccurrenceRate
      if ((_onsetCheckCounter > _minOccurrenceRateThreshold)){
        _NOTED_ON = true;
        if (note != _maxVoted[0]){  // avoid note slicing effect
            _noteOff = true;
            _noteOn = true;
            updateDnote();
            note = _maxVoted[0];
        }
        //E_INFO("off-onset(" << _maxVoted[0] << ", uncoherent & NOTED): " << _onsetCheckCounter << " - " << _minOccurrenceRateThreshold);
        _offsetCheckCounter = 0;
        _onsetCheckCounter = 0;
      }
    }
    // output the max-voted midi note to avoid unestable midi note numbers
    setOutputs(_maxVoted[0], _minOccurrenceRatePeriod, _minOccurrenceRatePeriod);
    return;
  }

  if (!hasCoherence() && !_NOTED_ON) {
    if (_maxVoted[1] > _minOccurrenceRate) {
      _onsetCheckCounter++;

      if (_onsetCheckCounter > _minOnsetCheckThreshold) {
        if (_maxVoted[0] != 0.0) {
          note = _maxVoted[0];
          _NOTED_ON = true;
          _noteOn = true;
          //E_INFO("onset(" << _maxVoted[0] << ", uncoherent & unNOTED)");
          _onsetCheckCounter = 0;
          _offsetCheckCounter = 0;
        }
      }
    }
    // output the max-voted midi note to avoid unestable midi note numbers
    setOutputs(_maxVoted[0], _minOnsetCheckPeriod, _minOffsetCheckPeriod);
    return;
  }
  // E_INFO("Compute() -END");
}

void Pitch2Midi::updateDnote() {
  dnote = note;
}
