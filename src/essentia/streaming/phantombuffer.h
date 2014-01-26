/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_PHANTOMBUFFER_H
#define ESSENTIA_PHANTOMBUFFER_H

#include <vector>
#include "multiratebuffer.h"
#include "../roguevector.h"
#include "../threading.h"
#include "../essentiautil.h"


namespace essentia {
namespace streaming {

class Window {
 public:
  int begin;
  int end;
  int turn;

  Window() : begin(0), end(0), turn(0) {}

  inline int total(int bufferSize) const {
    return turn*bufferSize + begin;
  }
};


/**
 * The PhantomBuffer class is an implementation of the MultiRateBuffer interface
 * that has a special zone at its end, called the phantom zone, which is also
 * replicated at the beginning of the buffer, so that we can always guarantee
 * that retrieving any number of samples lower than the phantom size can be done
 * on a contiguous zone in memory.
 *
 * @todo class should be thread-safe, but make sure it really is
 *
 * NB: we can only guarantee that availableFor* returns a least the size of the phantom buffer, not more
 *     we have to choose the size of the phantom zone carefully, or make it dynamically resizable
 */
template <typename T>
class PhantomBuffer : public MultiRateBuffer<T> {

 public:

  PhantomBuffer(SourceBase* parent, BufferUsage::BufferUsageType type) {
    _parent = parent;
    setBufferType(type);
  }

  void setBufferType(BufferUsage::BufferUsageType type) {
    BufferInfo buf;
    switch (type) {
    case BufferUsage::forSingleFrames:
      buf.size = 16;
      buf.maxContiguousElements = 0;
      break;

    case BufferUsage::forAudioStream:
      buf.size = 65536;
      buf.maxContiguousElements = 4096;
      break;

    case BufferUsage::forLargeAudioStream:
      buf.size = 524288;
      buf.maxContiguousElements = 131072;
      break;

    default:
      throw EssentiaException("Unknown buffer type");
    }

    setBufferInfo(buf);
  }

  BufferInfo bufferInfo() const {
    BufferInfo info;
    info.size = _bufferSize;
    info.maxContiguousElements = _phantomSize;
    return info;
  }

  void setBufferInfo(const BufferInfo& info) {
    _bufferSize = info.size;
    _phantomSize = info.maxContiguousElements;
    _buffer.resize(_bufferSize + _phantomSize);
  }

  PhantomBuffer(SourceBase* parent, int size, int phantomSize) :
    _parent(parent),
    _bufferSize(size),
    _phantomSize(phantomSize),
    _buffer(size + phantomSize) {
    // initialize views and all??
  }

  /**
   * @todo implement me if necessary
   */
  ~PhantomBuffer() {}

  const std::vector<T>& readView(ReaderID id) const;
  std::vector<T>& writeView() { return _writeView; }

  /**
   * This method tries to acquire the requested number of tokens. It returns true
   * on success, or false if there were not enough tokens available.
   */
  bool acquireForRead(ReaderID id, int requested);

  /**
   * This method tries to acquire the requested number of tokens. It returns true
   * on success, or false if there were not enough tokens available.
   */
  bool acquireForWrite(int requested);

  void releaseForWrite(int released);
  void releaseForRead(ReaderID id, int released);

  // @todo mutex-lock these functions or not??
  /**
   * Add a new reader and return its ID. The reader will start at the point
   * where the write window is currently located.
   * If @c startFromZero is true, the reader will then start at t = 0, meaning
   * that a reader that is connected after some tokens are produced will still
   * get those tokens.
   */
  ReaderID addReader(bool startFromZero = false);
  void removeReader(ReaderID id);

  int numberReaders() const;

  /**
   * WARNING: do only before starting to use buffer, for initial configuration
   * @todo find a way to be able to call this function at any time to allow
   *       dynamic resizing, even by locking everything. It doesn't matter if
   *       it takes too much time, as it should be called only very rarely
   */
  void resize(int size, int phantomSize) {
    _buffer.resize(size+phantomSize);
    _bufferSize = size;
    _phantomSize = phantomSize;
  }

  int totalTokensWritten() const {
    MutexLocker lock(mutex); NOWARN_UNUSED(lock);
    return _writeWindow.total(_bufferSize);
  }

  int totalTokensRead(ReaderID id) const {
    MutexLocker lock(mutex); NOWARN_UNUSED(lock);
    return _readWindow[id].total(_bufferSize);
  }

  const T& lastTokenProduced() const {
    MutexLocker lock(mutex); NOWARN_UNUSED(lock);
    if (_writeWindow.total(_bufferSize) == 0) {
      throw EssentiaException("Tried to call ::lastTokenProduced() on ", _parent->fullName(),
                              " which hasn't produced any token yet");
    }

    int idx = _writeWindow.begin;
    if (idx == 0) return _buffer[_bufferSize-1];
    return _buffer[idx-1];
  }

  void reset();

 protected:
  SourceBase* _parent;

  int _bufferSize, _phantomSize; // bufferSize does not include phantomSize
  std::vector<T> _buffer; // the buffer where data is stored
  // bufferSize must be > phantomSize in all cases

  Window _writeWindow;
  std::vector<Window> _readWindow;

  RogueVector<T> _writeView;
  std::vector<RogueVector<T> > _readView; // @todo CAREFUL WHEN COPYING ROGUEVECTOR...

  // threading-related & locking structures
  mutable Mutex mutex; // should be locked before any modification to the object

 protected:
  // this function is only here to make sure we do not overflow the window.turn variable
  // @todo we could make the class smarter by not counting the turns, but just knowing if
  // readers are on the same turn as the writer, or if they're one turn late (use a bool
  // isTurnOdd (or even))
  void resetTurns();

  void updateReadView(ReaderID id);
  void updateWriteView();

  // mutex should be locked before entering this function
  // make sure it doesn't overflow
  int availableForRead(ReaderID id) const;
  int availableForWrite(bool contiguous=true) const;

  // reposition pointer if we're in the phantom zone
  void relocateReadWindow(ReaderID id);
  void relocateWriteWindow();

};

} // namespace streaming
} // namespace essentia

#include "phantombuffer_impl.h"

#endif // ESSENTIA_PHANTOMBUFFER_H
