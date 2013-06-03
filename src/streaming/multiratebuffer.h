/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MULTIRATEBUFFER_H
#define ESSENTIA_MULTIRATEBUFFER_H

#include <vector>
#include "types.h"

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
  virtual int availableForWrite() const = 0;

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
