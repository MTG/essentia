/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_STREAMCONNECTOR_H
#define ESSENTIA_STREAMCONNECTOR_H


namespace essentia {
namespace streaming {

/**
 * This class represents a stream connector (end-point), which is the basic
 * interface for both Sinks and Sources. It provides methods for querying
 * how many tokens are available, acquire & release them.
 * It can be used both for input and output, as it only describes the data that
 * flows in a stream, without any indication of the direction in which it flows.
 */
class StreamConnector {
 public:

  StreamConnector() : _acquireSize(0), _releaseSize(0) {}

  virtual ~StreamConnector() {}

  /**
   * Returns how many tokens are available in the stream.
   */
  virtual int available() const = 0;

  /**
   * Acquire (consume) the default number of tokens.
   */
  inline bool acquire() { return acquire(_acquireSize); }

  /**
   * Acquire (consume) the requested number of tokens.
   */
  virtual bool acquire(int n) = 0;

  /**
   * Release (produce) the default number of tokens.
   */
  inline void release() { release(_releaseSize); }

  /**
   * Release (produce) the requested number of tokens.
   */
  virtual void release(int n) = 0;

  /**
   * Returns the default number of tokens to be acquired.
   */
  virtual int acquireSize() const { return _acquireSize; }

  /**
   * Returns the default number of tokens to be released.
   */
  virtual int releaseSize() const { return _releaseSize; }

  /**
   * Set the default number of tokens to be acquired.
   */
  virtual void setAcquireSize(int n) { _acquireSize = n; }

  /**
   * Set the default number of tokens to be released.
   */
  virtual void setReleaseSize(int n) { _releaseSize = n; }

  /**
   * Resets the state of this StreamConnector
   */
  virtual void reset() = 0;

 protected:
  int _acquireSize;
  int _releaseSize;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMCONNECTOR_H
