/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_MULTIRATEBUFFER_H
#define ESSENTIA_MULTIRATEBUFFER_H

#include <vector>
#include "../types.h"

namespace essentia {
namespace streaming {

template <typename T>
class MultiRateBuffer {

 public:
  virtual ~MultiRateBuffer() {}

  virtual void setBufferType(BufferUsage::BufferUsageType type) = 0;
  virtual BufferInfo bufferInfo() const = 0;
  virtual void setBufferInfo(const BufferInfo& info) = 0;

  // add/remove readers to/from the buffer
  // returns the id of the newly attached reader
  virtual ReaderID addReader(bool startFromZero = false) = 0;
  virtual void removeReader(ReaderID id) = 0;
  virtual int numberReaders() const = 0;

  // get and release access to the items in the buffer
  virtual bool acquireForRead(ReaderID id, int requested) = 0;
  virtual void releaseForRead(ReaderID id, int released) = 0;
  virtual bool acquireForWrite(int requested) = 0;
  virtual void releaseForWrite(int released) = 0;

  virtual int availableForRead(ReaderID id) const = 0;
  virtual int availableForWrite(bool contiguous=true) const = 0;

  virtual int totalTokensRead(ReaderID id) const = 0;
  virtual int totalTokensWritten() const = 0;

  virtual const T& lastTokenProduced() const = 0;

  // some useful aliases, depending on the terminology used
  void readerConsume(ReaderID id, int requested) { acquireForRead(id, requested); }
  void readerProduce(ReaderID id, int released) { releaseForRead(id, released); }
  void writerConsume(int requested) { acquireForWrite(requested); }
  void writerProduce(int released) { releaseForWrite(released); }

  // get views of the data currently being accessed to
  virtual const std::vector<T>& readView(ReaderID id) const = 0;
  virtual std::vector<T>& writeView() = 0;

  virtual void reset() = 0;

  // @todo remove this, only here for debug
  virtual void resize(int size, int phantomSize) = 0;

};

// it's the MultiRateBuffer that does all the allocation
// the source and sink do not allocate memory, they just take
// ownership of the memory zone in the buffer

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MULTIRATEBUFFER_H
