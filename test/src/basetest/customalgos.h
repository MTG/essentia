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

#ifndef CUSTOMALGOS_H
#define CUSTOMALGOS_H

#include "streamingalgorithm.h"
#include "streamingalgorithmcomposite.h"
#include "essentia_gtest.h"

namespace essentia {
namespace streaming {

class CompositeAlgo : public AlgorithmComposite {
 protected:
  Algorithm* _copy;

  SinkProxy<float> _srcProxy;
  SourceProxy<float> _destProxy;

 public:
  CompositeAlgo() {
    declareInput(_srcProxy, "src", "source");
    declareOutput(_destProxy, "dest", "dest");

    _copy = AlgorithmFactory::create("CopyAlgo");
    _srcProxy >> _copy->input("src");
    _copy->output("dest") >> _destProxy;
  }

  ~CompositeAlgo() { delete  _copy; }

  void declareParameters() {}

  void declareProcessOrder() {
    declareProcessStep(SingleShot(_copy));
    //declareProcessStep(ChainFrom(_copy));
  }

  AlgorithmStatus process() {
    // this is a hack
    return _copy->process();
  }


  void configure() {}

  void reset() {
    AlgorithmComposite::reset();

    // recreate the copy algorithm
    DBG("delete");
    delete _copy;
    _copy = AlgorithmFactory::create("CopyAlgo");
    DBG("reattach");
    _srcProxy >> _copy->input("src");
    _copy->output("dest") >> _destProxy;
  }

  static const char* name;
  static const char* version;
  static const char* description;

};

class DiamondShapeAlgo : public AlgorithmComposite {
 protected:
  Algorithm *_fcutter, *_spectrum, *_peaks, *_pitch, *_hpeaks;

  SinkProxy<Real> _srcProxy;
  SourceProxy<std::vector<Real> > _destProxy;

 public:
  DiamondShapeAlgo() {
    declareInput(_srcProxy, "src", "source");
    declareOutput(_destProxy, "dest", "dest");

    _fcutter  = AlgorithmFactory::create("FrameCutter");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _peaks    = AlgorithmFactory::create("SpectralPeaks");
    _pitch    = AlgorithmFactory::create("PitchYinFFT");
    _hpeaks   = AlgorithmFactory::create("HarmonicPeaks");

    _srcProxy >> _fcutter->input("signal");

    _fcutter->output("frame")     >> _spectrum->input("frame");
    _spectrum->output("spectrum") >> _peaks->input("spectrum");
    _spectrum->output("spectrum") >> _pitch->input("spectrum");
    _peaks->output("frequencies") >> _hpeaks->input("frequencies");
    _peaks->output("magnitudes")  >> _hpeaks->input("magnitudes");
    _pitch->output("pitch")       >> _hpeaks->input("pitch");

    _hpeaks->output("harmonicFrequencies") >> _destProxy;
    _hpeaks->output("harmonicMagnitudes")  >> NOWHERE;
  }

  ~DiamondShapeAlgo() {
    delete _fcutter;
    delete _spectrum;
    delete _peaks;
    delete _pitch;
    delete _hpeaks->output("harmonicMagnitudes").sinks()[0]->parent(); // NOWHERE
    delete _hpeaks;
  }

  void declareParameters() {}

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_fcutter));
  }

  static const char* name;
  static const char* version;
  static const char* description;

};

class CopyAlgo : public Algorithm {
 protected:
  Sink<float> _src;
  Source<float> _dest;

 public:
  CopyAlgo() {
    declareInput(_src, 1, "src", "where to copy from");
    declareOutput(_dest, 1, "dest", "where to copy to");
  }

  AlgorithmStatus process() {
    AlgorithmStatus status = acquireData();
    if (status != OK) return status;

    _dest.firstToken() = _src.firstToken();

    releaseData();

    return OK;
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;

};


class TeeAlgo : public Algorithm {
 protected:
  Sink<float> _src;
  Source<float> _dest1;
  Source<float> _dest2;

public:
  TeeAlgo() {
    declareInput(_src, 1, "src", "where to copy from");
    declareOutput(_dest1, 1, "dest1", "where to copy to");
    declareOutput(_dest2, 1, "dest2", "where to copy to");
  }

  AlgorithmStatus process() {
    AlgorithmStatus status = acquireData();
    if (status != OK) return status;

    _dest1.firstToken() = _src.firstToken();
    _dest2.firstToken() = _src.firstToken();

    releaseData();

    return OK;
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;

};

class TeeProxyAlgo : public AlgorithmComposite {
 protected:
  Algorithm *_tee;

  SinkProxy<Real> _srcProxy;
  SourceProxy<Real> _destProxy;

 public:
  TeeProxyAlgo() {
    declareInput(_srcProxy, "src", "source");
    declareOutput(_destProxy, "dest", "dest");

    _tee  = AlgorithmFactory::create("TeeAlgo");

    _srcProxy >> _tee->input("src");

    _tee->output("dest1") >> NOWHERE;
    _tee->output("dest2") >> _destProxy;
  }

  ~TeeProxyAlgo() {
    delete _tee;
  }

  void declareParameters() {}

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_tee));
  }

  static const char* name;
  static const char* version;
  static const char* description;

};


#define DECLARE_TEST_ALGO(AlgoName)                 \
class AlgoName : public Algorithm {                 \
protected:                                          \
  Sink<float> _input;                               \
  Source<float> _output;                            \
public:                                             \
  AlgoName() {                                      \
    declareInput(_input, 1, "in", "the input");     \
    declareOutput(_output, 1, "out", "the output"); \
  }                                                 \
                                                    \
  AlgorithmStatus process() {                       \
    return PASS;                                    \
  }                                                 \
                                                    \
  void declareParameters() {}                       \
                                                    \
  static const char* name;                          \
  static const char* version;                       \
  static const char* description;                   \
}

#define DEFINE_TEST_ALGO(AlgoName)                            \
const char* essentia::streaming::AlgoName::name = #AlgoName;  \
const char* essentia::streaming::AlgoName::version = "1.0";   \
const char* essentia::streaming::AlgoName::description = "";

#define REGISTER_ALGO(AlgoName) \
essentia::streaming::AlgorithmFactory::Registrar<essentia::streaming::AlgoName> reg##AlgoName;

DECLARE_TEST_ALGO(A);
DECLARE_TEST_ALGO(B);
DECLARE_TEST_ALGO(C);
DECLARE_TEST_ALGO(D);
DECLARE_TEST_ALGO(E);
DECLARE_TEST_ALGO(F);
DECLARE_TEST_ALGO(G);
DECLARE_TEST_ALGO(H);
DECLARE_TEST_ALGO(I);
DECLARE_TEST_ALGO(J);

DECLARE_TEST_ALGO(DevNullSample);
DECLARE_TEST_ALGO(PoolStorageFrame);



/*

---------- Algorithms needed for building the following network -----------

  +------------- A -------------+
  | +-----+   +--------------+  |     +---+
  | |     |---|      C       |--|--+--| G |
  | |     |   +--------------+  |     +---+
  | |     |                     |
  | |  B  |   +------ D -----+  |     +---+
  | |     |   | +---+  +---+ |  |     | H |
  | |     |   | |   |--|   | |  |    _+---+
  | |     |---|-| E |  | F |-|--|---<_
  | |     |   | |   |--|   | |  |     +---+
  | |     |   | +---+  +---+ |  |     | I |
  | +-----+   +--------------+  |     +---+
  +-----------------------------+

*/


class B1 : public Algorithm {
protected:
  Source<float> _output1;
  Source<float> _output2;
public:
  B1() {
    declareOutput(_output1, 1, "out1", "the output");
    declareOutput(_output2, 1, "out2", "the output");
  }

  AlgorithmStatus process() {
    return PASS;
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;
};

class E1 : public Algorithm {
protected:
  Sink<float> _input;
  Source<float> _output1;
  Source<float> _output2;
public:
  E1() {
    declareInput(_input, 1, "in", "the input");
    declareOutput(_output1, 1, "out1", "the output");
    declareOutput(_output2, 1, "out2", "the output");
  }

  AlgorithmStatus process() {
    return PASS;
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;
};

class F1 : public Algorithm {
protected:
  Sink<float> _input1;
  Sink<float> _input2;
  Source<float> _output;
public:
  F1() {
    declareInput(_input1, 1, "in1", "the input");
    declareInput(_input2, 1, "in2", "the input");
    declareOutput(_output, 1, "out", "the output");
  }

  AlgorithmStatus process() {
    return PASS;
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;
};

class D1 : public AlgorithmComposite {
 protected:
  SinkProxy<float> _in;
  SourceProxy<float> _out;

  Algorithm *_E, *_F;

 public:
  D1() {
    declareInput(_in, "in", "input");
    declareOutput(_out, "out", "output");

    _E = AlgorithmFactory::create("E1");
    _F = AlgorithmFactory::create("F1");

    connect(_E->output("out1"), _F->input("in1"));
    connect(_E->output("out2"), _F->input("in2"));

    attach(_in, _E->input("in"));
    attach(_F->output("out"), _out);
  }

  ~D1() {
    delete _E; delete _F;
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_E));
  }

  void declareParameters() {}

  static const char* name;
  static const char* version;
  static const char* description;
};


#define DECLARE_COMPOSITE_A(AlgoName)              \
class AlgoName : public AlgorithmComposite {       \
 protected:                                        \
  SourceProxy<float> _out1;                        \
  SourceProxy<float> _out2;                        \
                                                   \
  Algorithm *_B, *_C, *_D;                         \
                                                   \
 public:                                           \
  AlgoName() {                                     \
    declareOutput(_out1, "out1", "output 1");      \
    declareOutput(_out2, "out2", "output 2");      \
                                                   \
    _B = AlgorithmFactory::create("B1");           \
    _C = AlgorithmFactory::create("C");            \
    _D = AlgorithmFactory::create("D1");           \
                                                   \
    connect(_B->output("out1"), _C->input("in"));  \
    connect(_B->output("out2"), _D->input("in"));  \
                                                   \
    attach(_C->output("out"), _out1);              \
    attach(_D->output("out"), _out2);              \
  }                                                \
                                                   \
  ~AlgoName() {                                    \
    delete _B; delete _C; delete _D;               \
  }                                                \
                                                   \
  void declareProcessOrder() {

#define DECLARE_COMPOSITE_END()                    \
  }                                                \
                                                   \
  void declareParameters() {}                      \
                                                   \
  static const char* name;                         \
  static const char* version;                      \
  static const char* description;                  \
};


DECLARE_COMPOSITE_A(A1)
declareProcessStep(ChainFrom(_B));
DECLARE_COMPOSITE_END()

DECLARE_COMPOSITE_A(A2)
declareProcessStep(ChainFrom(_B));
declareProcessStep(SingleShot(this));
DECLARE_COMPOSITE_END()

DECLARE_COMPOSITE_A(A3)
declareProcessStep(ChainFrom(_B));
declareProcessStep(SingleShot(_D));
DECLARE_COMPOSITE_END()

DECLARE_COMPOSITE_A(A4)
declareProcessStep(ChainFrom(_B));
declareProcessStep(SingleShot(this));
declareProcessStep(ChainFrom(_D));
declareProcessStep(SingleShot(this));
DECLARE_COMPOSITE_END()


#define DECLARE_COMPOSITE_Abis(AlgoName)           \
class AlgoName : public AlgorithmComposite {       \
 protected:                                        \
  SinkProxy<float> _in;                            \
  SourceProxy<float> _out1;                        \
  SourceProxy<float> _out2;                        \
  SourceProxy<float> _out3;                        \
                                                   \
  Algorithm *_B, *_C, *_E, *_F, *_G;               \
                                                   \
 public:                                           \
  AlgoName() {                                     \
    declareInput(_in, "in", "input");              \
    declareOutput(_out1, "out1", "output 1");      \
    declareOutput(_out2, "out2", "output 2");      \
    declareOutput(_out3, "out3", "output 3");      \
                                                   \
    _B = AlgorithmFactory::create("B");            \
    _C = AlgorithmFactory::create("C");            \
    _E = AlgorithmFactory::create("E1");           \
    _F = AlgorithmFactory::create("F");            \
    _G = AlgorithmFactory::create("G");            \
                                                   \
    connect(_E->output("out1"), _F->input("in"));  \
    connect(_E->output("out2"), _G->input("in"));  \
    connect(_B->output("out"), _C->input("in"));   \
                                                   \
    attach(_in, _E->input("in"));                  \
    attach(_F->output("out"), _out1);              \
    attach(_G->output("out"), _out2);              \
    attach(_C->output("out"), _out3);              \
  }                                                \
                                                   \
  ~AlgoName() {                                    \
    delete _B; delete _C; delete _E;               \
    delete _F; delete _G;                          \
  }                                                \
                                                   \
  void declareProcessOrder() {

DECLARE_COMPOSITE_Abis(A5)
declareProcessStep(ChainFrom(_E));
declareProcessStep(ChainFrom(_B));
DECLARE_COMPOSITE_END()

DECLARE_COMPOSITE_Abis(A6)
declareProcessStep(ChainFrom(_B));
declareProcessStep(ChainFrom(_E));
DECLARE_COMPOSITE_END()


} // namespace streaming
} // namespace essentia

#endif // CUSTOMALGOS_H
