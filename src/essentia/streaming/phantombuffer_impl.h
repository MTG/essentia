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

#ifndef ESSENTIA_PHANTOMBUFFER_IMPL_H
#define ESSENTIA_PHANTOMBUFFER_IMPL_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

template <typename T>
const std::vector<T>& PhantomBuffer<T>::readView(ReaderID id) const {
  return _readView[id];
}


template <typename T>
ReaderID PhantomBuffer<T>::addReader(bool startFromZero) {
  // add read window & view, just at where our write window is
  Window w;
  if (!startFromZero) {
    w.end = w.begin = _writeWindow.begin;
  }
  _readWindow.push_back(w);

  ReaderID id = _readWindow.size() - 1; // index of last one

  _readView.push_back(RogueVector<T>());
  updateReadView(id);

  return id;
}

template <typename T>
void PhantomBuffer<T>::removeReader(ReaderID id) {
  _readView.erase(_readView.begin() + id);
  _readWindow.erase(_readWindow.begin() + id);
}


template <typename T>
int PhantomBuffer<T>::numberReaders() const {
  return _readWindow.size();
}


/**
 * This method tries to acquire the requested number of tokens. It returns true
 * on success, or false if there were not enough tokens available.
 */
template <typename T>
bool PhantomBuffer<T>::acquireForRead(ReaderID id, int requested) {

  //DEBUG_NL("acquire " << requested << " for read (id: " << id << "), (" << availableForRead(id) << " available)");

  // we can afford to have phantomSize + 1 here, because either:
  // 1) we're strictly before the phantom zone (from at least 1 token), so no pb
  // 2) we're just at the beginning of the phantom zone, but in that case we
  //    should have been relocated to the beginning of the buffer
  if (requested > (_phantomSize + 1)) {
    // warning: this could cause a buffer to block, we need to reallocate or throw an exception here
    std::ostringstream msg;
    msg << "acquireForRead: Requested number of tokens (" << requested << ") > phantom size (" << _phantomSize << ")";
    msg << " in " << _parent->fullName() << " → " << _parent->sinks()[id]->fullName();
    throw EssentiaException(msg);
  }

  MutexLocker lock(mutex); NOWARN_UNUSED(lock);
  if (availableForRead(id) < requested) return false;

  _readWindow[id].end = _readWindow[id].begin + requested;
  updateReadView(id);

  return true;
}

/**
 * This method acquires (reserves) the requested number of tokens for the Writer
 * (Source). If not enough tokens were available, it puts the thread to sleep
 * and waits until the freeSpaceAvailable condition has been signalled, or until
 * it times out.
 */
template <typename T>
bool PhantomBuffer<T>::acquireForWrite(int requested) {

  //DEBUG_NL("acquire " << requested << " for write... (" << availableForWrite() << " available)");

  if (requested > (_phantomSize + 1)) {
    // warning: this could cause a buffer to block, we need to reallocate or throw an exception here
    std::ostringstream msg;
    msg << "acquireForWrite: Requested number of tokens (" << requested << ") > phantom size (" << _phantomSize << ")";
    msg << " in " << _parent->fullName();
    throw EssentiaException(msg);
  }

  MutexLocker lock(mutex); NOWARN_UNUSED(lock);
  if (availableForWrite() < requested) return false;

  _writeWindow.end = _writeWindow.begin + requested;
  updateWriteView();

  return true;
}

template <typename T>
void PhantomBuffer<T>::releaseForWrite(int released) {
  MutexLocker lock(mutex); NOWARN_UNUSED(lock);

  // error checking:
  if (released > _writeWindow.end - _writeWindow.begin) {
    std::ostringstream msg;
    msg << _parent->fullName() << ": releasing too many tokens (write access): "
        << released << " instead of " << _writeWindow.end - _writeWindow.begin << " max allowed";
    throw EssentiaException(msg);
  }

  // replicate from the beginning to the phantom zone if necessary
  if (_writeWindow.begin < _phantomSize) {
    T* first  = &_buffer[_writeWindow.begin];
    T* last   = &_buffer[std::min(_writeWindow.begin + released, _phantomSize)];
    T* result = &_buffer[_writeWindow.begin + _bufferSize];
    fastcopy(result, first, last-first);
  }
  // replicate from the phantom zone to the beginning if necessary
  else if (_writeWindow.end > _bufferSize) {
    int beginIdx = std::max(_writeWindow.begin, (int)_bufferSize);
    T* first  = &_buffer[beginIdx];
    T* last   = &_buffer[_writeWindow.end];
    T* result = &_buffer[beginIdx - _bufferSize];
    fastcopy(result, first, last-first);
  }

  _writeWindow.begin += released;
  relocateWriteWindow();
  updateWriteView();

  //DEBUG_NL(" - total written tokens: " << _writeWindow.total(_bufferSize));
}

template <typename T>
void PhantomBuffer<T>::releaseForRead(ReaderID id, int released) {
  MutexLocker lock(mutex); NOWARN_UNUSED(lock);
  Window& w = _readWindow[id];

  // error checking:
  if (released > w.end - w.begin) {
    std::ostringstream msg;
    msg << _parent->fullName() << ": releasing too many tokens (read access): "
        << released << " instead of " << w.end - w.begin << " max allowed";
    throw EssentiaException(msg);
  }

  w.begin += released;
  relocateReadWindow(id);
  updateReadView(id);

  //DEBUG_NL(" - total read tokens: " << w.total(_bufferSize));
}


////////// -- protected methods implementation


template <typename T>
inline void PhantomBuffer<T>::resetTurns() {
  // only do it when necessary
  if (_writeWindow.turn < 1000000)
    return;

  // get maximum number of turns we can substract
  int m = _writeWindow.turn;

  for (uint i=0; i<_readWindow.size(); i++) {
    m = std::min(m, _readWindow[i].turn);
  }
  // substract turns
  _writeWindow.turn -= m;
  for (uint i=0; i<_readWindow.size(); i++) {
    _readWindow[i].turn -= m;
  }
}

template <typename T>
inline void PhantomBuffer<T>::updateReadView(ReaderID id) {
  const RogueVector<T>& vconst = static_cast<const RogueVector<T>&>(readView(id));
  RogueVector<T>& v = const_cast<RogueVector<T>&>(vconst);
  v.setData(&_buffer[0] + _readWindow[id].begin);
  v.setSize(_readWindow[id].end - _readWindow[id].begin);
}

template <typename T>
inline void PhantomBuffer<T>::updateWriteView() {
  _writeView.setData(&_buffer[0] + _writeWindow.begin);
  _writeView.setSize(_writeWindow.end - _writeWindow.begin);
}


// mutex should be locked before entering this function
// make sure it doesn't overflow
/**
 * This method computes the maximum number of contiguous tokens that can be
 * acquired by the Reader at this moment. It is computed as the minimum between
 * the theoretical number of tokens available (without the contiguous condition)
 * and the number of contiguous tokens from the place where we are inside the
 * buffer.
 */
template <typename T>
int PhantomBuffer<T>::availableForRead(ReaderID id) const {
  //relocateReadWindow(id); // this call should be useless, but it's a safety guard to have it

  int theoretical = _writeWindow.total(_bufferSize) - _readWindow[id].total(_bufferSize);
  int contiguous = _bufferSize + _phantomSize - _readWindow[id].begin;

  /*
  DEBUG_NL("avail for read: " << _readWindow[id].total(_bufferSize)
        << " write: " << _writeWindow.total(_bufferSize)
        << " final: " << min(theoretical, contiguous));
  */

  return std::min(theoretical, contiguous);
}

/**
 * This method computes the maximum number of contiguous tokens that can be
 * acquired by the Writer at this moment. It is computed as the minimum between
 * the theoretical number of tokens available (without the contiguous condition)
 * and the number of contiguous tokens from the place where we are inside the
 * buffer.
 */
template <typename T>
int PhantomBuffer<T>::availableForWrite(bool contiguous) const {
  //relocateWriteWindow(); // this call should be useless, but it's a safety guard to have it

  int minTotal = _bufferSize;
  if (!_readWindow.empty()) { // someone is connected, take its value instead of bufferSize
    minTotal = _readWindow.begin()->total(_bufferSize);
  }

  //DEBUG_PLAIN(_writeWindow.total(_bufferSize) << " read:");

  // for each read window, find the one that is the latest, as it is the one
  // that the write window should not overtake.
  for (uint i=0; i<_readWindow.size(); i++) {
    const Window& w = _readWindow[i];
    minTotal = std::min(minTotal, w.total(_bufferSize));
  }

  int theoretical = minTotal - _writeWindow.total(_bufferSize) + _bufferSize;
  if (!contiguous) {
    return theoretical;
  }

  int ncontiguous = _bufferSize + _phantomSize - _writeWindow.begin;
  return std::min(theoretical, ncontiguous);
}

// reposition pointer if we're in the phantom zone
template <typename T>
void PhantomBuffer<T>::relocateWriteWindow() {
  if (_writeWindow.begin >= _bufferSize) {
    _writeWindow.begin -= _bufferSize;
    _writeWindow.end -= _bufferSize;
    _writeWindow.turn++;
    //resetTurns();
  }
}

// reposition pointer if we're in the phantom zone
template <typename T>
void PhantomBuffer<T>::relocateReadWindow(ReaderID id) {
  Window& w = _readWindow[id];
  if (w.begin >= _bufferSize) {
    w.begin -= _bufferSize;
    w.end -= _bufferSize;
    w.turn++;
    //resetTurns();
  }
}

template <typename T>
void PhantomBuffer<T>::reset() {
  // we don't need to clear the buffer, because when new data is written to the
  // buffer, it will overwrite the old data, and no one can read the old data
  // until new data is written
  //_buffer.clear();
  _writeWindow = Window();
  for (int i=0; i<(int)_readWindow.size(); i++) {
    _readWindow[i] = Window();
  }
}

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PHANTOMBUFFER_IMPL_H
