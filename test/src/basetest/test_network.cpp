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

#include "essentia_gtest.h"
#include "network.h"
#include "networkparser.h"
#include "graphutils.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


inline string getName(NetworkNode* node) {
  return node->algorithm()->name();
}

inline Algorithm* getAlgo(NetworkNode* node) {
  return node->algorithm();
}

TEST(Network, SimpleVisibleNetwork) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* loader = factory.create("MonoLoader",
                                     "filename", "test/audio/recorded/cat_purrrr.wav");

  Algorithm* lowpass = factory.create("LowPass");

  loader->output("audio")    >>  lowpass->input("signal");
  lowpass->output("signal")  >>  NOWHERE;

  Network network(loader);

  const char* expected[] = {
    "+------------+  +----------+  +---------+",
    "| MonoLoader |--| LowPass  |--| DevNull |",
    "+------------+  +----------+  +---------+",
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(network.visibleNetworkRoot(),
                                        VISIBLE_NETWORK(expected)));
}

TEST(Network, SimpleExecutionNetwork) {
  const char* composite[] = {
    "+------------+",
    "| MonoLoader |",
    "+------------+",
  };

  const char* expanded[] = {
    " +---------------+    +-------------+   +------------+  ",
    " |  AudioLoader  |----|  MonoMixer  |---|  Resample  |  ",
    " +---------------+    +-------------+   +------------+  "
  };

  // create the expanded network by hand because AudioLoader and MonoMixer don't
  // have the same number of inputs/outputs
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* loader   = factory.create("AudioLoader");
  Algorithm* mixer    = factory.create("MonoMixer");
  Algorithm* resample = factory.create("Resample");

  loader->output("audio")           >>  mixer->input("audio");
  loader->output("numberChannels")  >>  mixer->input("numberChannels");
  mixer->output("audio")            >>  resample->input("signal");

  Network n1(loader);
  NetworkParser np2(expanded, false);

  ASSERT_TRUE(areNetworkTopologiesEqual(n1.executionNetworkRoot(),
                                        np2.network()->visibleNetworkRoot()));

  ASSERT_TRUE(areNetworkTopologiesEqual(NetworkParser(composite).network()->executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}



TEST(Network, ExecutionNetworkWithComposite) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* loader = factory.create("MonoLoader",
                                     "filename", "test/audio/recorded/cat_purrrr.wav");

  Algorithm* lowpass = factory.create("LowPass");

  loader->output("audio")    >>  lowpass->input("signal");
  lowpass->output("signal")  >>  NOWHERE;

  Network network(loader);

  const char* expected[] = {
   " +---------------+    +-------------+   +------------+  +----------+  +---------+",
   " |  AudioLoader  |----|  MonoMixer  |---|  Resample  |--| LowPass  |--| DevNull |",
   " +---------------+    +-------------+   +------------+  +----------+  +---------+"
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(network.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expected)));

}

TEST(Network, ExecutionNetworkKeyExtractor) {
  const char* extractor[] = {
    "+------------+  +--------------+",
    "| MonoLoader |--| KeyExtractor |",
    "+------------+  +--------------+",
  };

  const char* expanded[] = {
    "                 MonoLoader                                                            ",
    "  <------------------------------------------------>                                   ",
    "                                                                                       ",
    " +---------------+    +-------------+   +------------+                                 ",
    " |  AudioLoader  |----|  MonoMixer  |---|  Resample  |--+                              ",
    " +---------------+    +-------------+   +------------+  |                              ",
    "                                                        |                              ",
    "   +----------------------------------------------------+                              ",
    "   |                                                                                   ",
    "   |  <------------------------ KeyExtractor ------------------------------>           ",
    "   |                                                                                   ",
    "   |   +---------------+    +-------------+   +------------+                           ",
    "   +---|  FrameCutter  |----|  Windowing  |---|  Spectrum  |--+                        ",
    "       +---------------+    +-------------+   +------------+  |                        ",
    "                                                              |                        ",
    "         +----------------------------------------------------+                        ",
    "         |                                                                             ",
    "         | +---------------+    +-------+   +-------------+    +-------+               ",
    "         +-| SpectralPeaks |----| HPCP  |---| PoolStorage |----|  Key  |               ",
    "           +---------------+    +-------+   +-------------+    +-------+               ",
    "                                                                                       ",
    "                                             <---------- Key --------->                ",
    "                                                                                       "
  };

  NetworkParser np(extractor);

  ASSERT_TRUE(areNetworkTopologiesEqual(np.network()->executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}

TEST(Network, ExecutionNetworkKeyExtractorConnected) {
  const char* extractor[] = {
    "+------------+  +--------------+",
    "| MonoLoader |--| KeyExtractor |",
    "+------------+  +--------------+",
  };

  const char* expanded[] = {
    "                 MonoLoader                                                            ",
    "  <------------------------------------------------>                                   ",
    "                                                                                       ",
    " +---------------+    +-------------+   +------------+                                 ",
    " |  AudioLoader  |----|  MonoMixer  |---|  Resample  |--+                              ",
    " +---------------+    +-------------+   +------------+  |                              ",
    "                                                        |                              ",
    "   +----------------------------------------------------+                              ",
    "   |                                                                                   ",
    "   |  <------------------------ KeyExtractor ------------------------------>           ",
    "   |                                                                                   ",
    "   |   +---------------+    +-------------+   +------------+                           ",
    "   +---|  FrameCutter  |----|  Windowing  |---|  Spectrum  |--+                +------------+  ",
    "       +---------------+    +-------------+   +------------+  |           +----|  DevNull   |  ",
    "                                                              |           |    +------------+  ",
    "         +----------------------------------------------------+           |                    ",
    "         |                                                                |    +------------+  ",
    "         | +---------------+    +-------+   +-------------+    +-------+  +----|  DevNull   |  ",
    "         +-| SpectralPeaks |----| HPCP  |---| PoolStorage |----|  Key  |--+    +------------+  ",
    "           +---------------+    +-------+   +-------------+    +-------+  |                    ",
    "                                                                          |    +------------+  ",
    "                                             <---------- Key --------->   +----|  DevNull   |  ",
    "                                                                               +------------+  ",
  };

  NetworkParser np(extractor);

  Algorithm* keyExtractor = np.network()->findAlgorithm("KeyExtractor");
  keyExtractor->output("key")       >>  NOWHERE;
  keyExtractor->output("scale")     >>  NOWHERE;
  keyExtractor->output("strength")  >>  NOWHERE;
  np.network()->update();

  ASSERT_TRUE(areNetworkTopologiesEqual(np.network()->executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}



/*


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

void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

Should return:

   C - G
  /
 B       H
  \     /
   E - F
        \
         I

Why:

 All composites should be expanded into their respective trees

*/
TEST(Network, Complex1) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A1");
  Algorithm* G = factory.create("G");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");

  A->output("out1") >> G->input("in");
  A->output("out2") >> H->input("in");
  A->output("out2") >> I->input("in");

  Network n(A);

  const char* expanded[] = {
    "             +----+    +----+                 ",
    "          +--| C  |----| G  |                 ",
    "          |  +----+    +----+                 ",
    " +-----+  |                          +-----+  ",
    " | B1  |--+                       +--|  H  |  ",
    " +-----+  |                       |  +-----+  ",
    "          | +-----+    +-----+    |           ",
    "          +-|  E1 |----| F1  |----+           ",
    "            +-----+    +-----+    |  +-----+  ",
    "                                  +--|  I  |  ",
    "                                     +-----+  ",
    "                                              "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}

/*

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


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
  declareProcessStep(SingleShot(this));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

Should return:

   - C -   - G
  /     \ /
 B       A - H
  \     / \
   E - F   - I

Why:

 SingleShot(A) depends on the complete execution of ChainFrom(B), so it
 needs to come after all of B's leaves

*/
TEST(Network, Complex2) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A2");
  Algorithm* G = factory.create("G");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");

  A->output("out1") >> G->input("in");
  A->output("out2") >> H->input("in");
  A->output("out2") >> I->input("in");

  Network n(A);

  const char* expanded[] = {
    "                 +----+                          +----+       ",
    "          +------| C  |---------+            +---| G  |       ",
    "          |      +----+         |            |   +----+       ",
    " +-----+  |                     |   +----+   |                ",
    " | B1  |--+                     +---| A2 |---+   +----+       ",
    " +-----+  |                     |   +----+   +---| H  |       ",
    "          | +-----+    +-----+  |            |   +----+       ",
    "          +-|  E1 |----| F1  |--+            |                ",
    "            +-----+    +-----+               |   +----+       ",
    "                                             +---| I  |       ",
    "                                                 +----+       ",
    "                                                              "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}



/*

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


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
  declareProcessStep(SingleShot(D));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

Should return:

   - C -       - G
  /     \     /
 B       E - F - H
  \     /     \
   E - F       - I

Why:
 SingleShot(A) depends on the complete execution of ChainFrom(B), so it
 needs to come after all of B's leaves


Or the smartass way:

   - C - G
  /
 B               H
  \             /
   E - F - E - F
                \
                 I


Why:
 SingleShot(D) does not depend on C so we don't need to merge the branches

 Note: this could still be wrong, because G might depend on D, even though they're not connected
       this could happen if D::process() does some stuff to C, like reconfigure it, or call some
       functions on it because it has a pointer to it.
       Even though this cannot be seen in the topology, and would be really bad design, it is
       possible and as such we have to choose whether we want to have correctness in all the cases,
       at the expense (maybe) of performance, or do we want to hope for the best and assume that
       if 2 algorithms are not explicitly connected, then they do not interact in any way.

*/
TEST(Network, Complex3) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A3");
  Algorithm* G = factory.create("G");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");

  A->output("out1") >> G->input("in");
  A->output("out2") >> H->input("in");
  A->output("out2") >> I->input("in");

  Network n(A);

  const char* expanded[] = {
    "                 +----+                                   +----+       ",
    "          +------| C  |---------+                     +---| G  |       ",
    "          |      +----+         |                     |   +----+       ",
    " +-----+  |                     |   +----+   +----+   |                ",
    " | B1  |--+                     +---| E1 |---| F1 |---+   +----+       ",
    " +-----+  |                     |   +----+   +----+   +---| H  |       ",
    "          | +-----+    +-----+  |                     |   +----+       ",
    "          +-|  E1 |----| F1  |--+                     |                ",
    "            +-----+    +-----+                        |   +----+       ",
    "                                                      +---| I  |       ",
    "                                                          +----+       ",
    "                                                                       "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}



/*

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


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
  declareProcessStep(SingleShot(this, STEP_1));
  declareProcessStep(SingleShot(D));
  declareProcessStep(SingleShot(this, STEP_2));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

Should return:


   - C -               - G
  /     \             /
 B       A - E - F - A - H
  \     /             \
   E - F               - I

*/
TEST(Network, Complex4) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A4");
  Algorithm* G = factory.create("G");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");

  A->output("out1") >> G->input("in");
  A->output("out2") >> H->input("in");
  A->output("out2") >> I->input("in");

  Network n(A);

  const char* expanded[] = {
    "                 +----+                                                        +----+  ",
    "          +------| C  |---------+                                          +---| G  |  ",
    "          |      +----+         |                                          |   +----+  ",
    " +-----+  |                     |   +----+    +----+     +----+   +----+   |           ",
    " | B1  |--+                     +---| A4 |----| E1 |-----| F1 |---| A4 |---+   +----+  ",
    " +-----+  |                     |   +----+    +----+     +----+   +----+   +---| H  |  ",
    "          | +-----+    +-----+  |                                          |   +----+  ",
    "          +-|  E1 |----| F1  |--+                                          |           ",
    "            +-----+    +-----+                                             |   +----+  ",
    "                                                                           +---| I  |  ",
    "                                                                               +----+  ",
    "                                                                                       "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}



/*

            +-------- A ---------+
            |            +---+   |    +---+
            |         +--| F |---+----| H |
  +---+     |  +---+  |  +---+   |    +---+
  | D |-----+--| E |--+          |
  +---+     |  +---+  |  +---+   |    +---+
            |         +--| G |---+----| I |
            |            +---+   |    +---+
            |                    |
            |  +---+     +---+   |    +---+
            |  | B |-----| C |---+----| J |
            |  +---+     +---+   |    +---+
            +--------------------+


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
  declareProcessStep(ChainFrom(B));
}

Should return:


       - F -       - H
      /     \     /
 D - E       B - C - I
      \     /     \
       - G -       - J

Why:
 ChainFrom(B) comes *after* ChainFrom(E), so we cannot assume anymore
 that F → H and G → I, but we have to make them depend on the full completion
 of the last chain, which occurs after C.

*/
TEST(Network, Complex5) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A5");
  Algorithm* D = factory.create("D");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");
  Algorithm* J = factory.create("J");

  D->output("out") >> A->input("in");
  A->output("out1") >> H->input("in");
  A->output("out2") >> I->input("in");
  A->output("out3") >> J->input("in");

  Network n(D);

  const char* expanded[] = {
    "                       +----+                          +----+   ",
    "                   +---| F  |--+                    +--| H  |   ",
    "                   |   +----+  |                    |  +----+   ",
    "                   |           |                    |           ",
    " +----+    +----+  |           |  +----+    +----+  |  +----+   ",
    " | D  |----| E1 |--+           +--| B  |----| C  |--+--| I  |   ",
    " +----+    +----+  |           |  +----+    +----+  |  +----+   ",
    "                   |           |                    |           ",
    "                   |   +----+  |                    |  +----+   ",
    "                   +---| G  |--+                    +--| J  |   ",
    "                       +----+                          +----+   ",
    "                                                                "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}


/*

            +-------- A ---------+
            |            +---+   |    +---+
            |         +--| F |---+----| H |
  +---+     |  +---+  |  +---+   |    +---+
  | D |-----+--| E |--+          |
  +---+     |  +---+  |  +---+   |    +---+
            |         +--| G |---+----| I |
            |            +---+   |    +---+
            |                    |
            |  +---+     +---+   |    +---+
            |  | B |-----| C |---+----| J |
            |  +---+     +---+   |    +---+
            +--------------------+


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
  declareProcessStep(ChainFrom(E));
}

Should return:

                   - H
                  /
               - F -
              /     \
 D - B - C - E       J
              \     /
               - G -
                  \
                   - I


Why:
 F → H and G → I are valid, because they occured during the last chain.
 However C → J is not valid anymore because C has not been visited in the last
 chain, so we must make J depend on the full completion of the last chain
 (last process step, really), which in this case would imply that J depends on
 both F and G which are the leaves of the last step, hence the diamond shape.


Note: do we want to accept this use case, or should we forbid it?
      A not-so-unreasonable condition would be that all inputs of a composite be
      connected to the first scheduled algorithm; but would this prevent us from
      having some real use case?

      See also use cases below for examples of configurations for which it is not
      clear whether we should allow them or not.

*/
TEST(Network, Complex6) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A6");
  Algorithm* D = factory.create("D");
  Algorithm* H = factory.create("H");
  Algorithm* I = factory.create("I");
  Algorithm* J = factory.create("J");

  D->output("out")  >> A->input("in");
  A->output("out1") >> H->input("in");
  A->output("out2") >> I->input("in");
  A->output("out3") >> J->input("in");

  Network n(D);

  const char* expanded[] = {
    "                                                               ",
    "                                                     +----+    ",
    "                                                 +---| H  |    ",
    "                                        +----+   |   +----+    ",
    "                                     +--| F  |---+             ",
    "                                     |  +----+   |   +----+    ",
    "   +----+   +----+   +----+   +----+ |           +---| J  |    ",
    "   | D  |---| B  |---| C  |---| E1 |-+               |    |    ",
    "   +----+   +----+   +----+   +----+ |           +---|    |    ",
    "                                     |  +----+   |   +----+    ",
    "                                     +--| G  |---+             ",
    "                                        +----+   |   +----+    ",
    "                                                 +---| I  |    ",
    "                                                     +----+    ",
    "                                                               "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}


/**
 * Note: diamond shapes in the visible topology of a network are normally not allowed, but
 *       there are places where we do use them, such as in:
 *
 *           /----- Pitch  ----\
 *   Spectrum                   HarmonicPeaks
 *           \- SpectralPeaks -/
 *
 * We have to make sure that at least this case where there are no composites involved works
 * without segfaulting or doing anything really stupid like that.
 */
TEST(Network, DiamondShape) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A");
  Algorithm* B = factory.create("B");
  Algorithm* C = factory.create("C");
  Algorithm* F = factory.create("F1");

  A->output("out") >> B->input("in");
  A->output("out") >> C->input("in");
  B->output("out") >> F->input("in1");
  C->output("out") >> F->input("in2");

  Network n(A);

  const char* expanded[] = {
    "                                         ",
    "                  +----+                 ",
    "              +---| B  |---+             ",
    "     +----+   |   +----+   |  +----+     ",
    "     | A  |---+            +--| F1 |     ",
    "     +----+   |   +----+   |  +----+     ",
    "              +---| C  |---+             ",
    "                  +----+                 ",
    "                                         "
  };

  const char* expandedWrong[] = {
    "                                         ",
    "                  +----+         +----+  ",
    "              +---| B  |---------| F1 |  ",
    "     +----+   |   +----+         +----+  ",
    "     | A  |---+                          ",
    "     +----+   |   +----+         +----+  ",
    "              +---| C  |---------| F1 |  ",
    "                  +----+         +----+  ",
    "                                         "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.visibleNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));

  ASSERT_FALSE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                         VISIBLE_NETWORK(expandedWrong)));

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));

}

TEST(Network, DiamondShape2) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* A = factory.create("A");
  // real-world use case, this corresponds to HarmonicPeaks
  Algorithm* DiamondShape = factory.create("DiamondShapeAlgo");
  Pool pool;

  A->output("out") >> DiamondShape->input("src");
  DiamondShape->output("dest") >> PC(pool, "freqs");

  Network n(A);
  n.update();

  vector<Algorithm*> order = n.linearExecutionOrder();

  ASSERT_EQ((size_t)8, order.size());
  EXPECT_EQ("A", order[0]->name());
  EXPECT_EQ("FrameCutter", order[1]->name());
  EXPECT_EQ("Spectrum", order[2]->name());
  EXPECT_TRUE((order[3]->name() == "PitchYinFFT" && order[4]->name() == "SpectralPeaks") ||
              (order[3]->name() == "SpectralPeaks" && order[4]->name() == "PitchYinFFT"));
  EXPECT_EQ("HarmonicPeaks", order[5]->name());
  EXPECT_TRUE((order[6]->name() == "PoolStorage" && order[7]->name() == "DevNull<std::vector<Real>>[0]") ||
              (order[6]->name() == "DevNull<std::vector<Real>>[0]" && order[7]->name() == "PoolStorage"));
}


TEST(Network, TeeProxyComposite) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* tp = factory.create("TeeProxyAlgo");
  Pool pool;

  tp->output("dest") >> NOWHERE;

  Network n(tp);


  const char* expanded[] = {
    "                                      ",
    "                        +----------+  ",
    "                    +---| DevNull  |  ",
    "     +----------+   |   +----------+  ",
    "     | TeeAlgo  |---+                 ",
    "     +----------+   |   +----------+  ",
    "                    +---| DevNull  |  ",
    "                        +----------+  ",
    "                                      "
  };

  ASSERT_TRUE(areNetworkTopologiesEqual(n.executionNetworkRoot(),
                                        VISIBLE_NETWORK(expanded)));
}

/*
--------------------------------------------------------------------------------

Note: the following is not allowed at the moment in Essentia, but it would be
      nice to be able to have it in the future (or maybe not?)


  +------------- A -------------+     +----- G ------+
  | +-----+   +--------------+  |     | +---+  +---+ |
  | |     |---|      C       |--|--+--|-| H |--|   | |
  | |     |   +--------------+  |     | +---+  |   | |
  | |     |                     |     |        |   | |
  | |  B  |   +------ D -----+  |     |        | J |-|--+
  | |     |   | +---+  +---+ |  |     |        |   | |
  | |     |   | |   |--|   | |  |     | +---+  |   | |
  | |     |---|-| E |  | F |-|--|--+--|-| I |--|   | |
  | |     |   | |   |--|   | |  |     | +---+  +---+ |
  | |     |   | +---+  +---+ |  |     +--------------+
  | +-----+   +--------------+  |
  +-----------------------------+


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

void G::declareProcessOrder() {
  declareProcessStep(SingleShot(H));
  declareProcessStep(ChainFrom(I));
}


Should return:

   - C -
  /     \
 B       H - I - J
  \     /
   E - F

Why:
 We're entering a new composite, so we depend on all its inputs as we cannot assume
 there will be no dependencies inside

or the smartass (correct?) way:

   - C - H -
  /         \
 B           J
  \         /
   E - F - I



  +------------- A -------------+     +----- G ------+
  | +-----+   +--------------+  |     | +---+  +---+ |
  | |     |---|      C       |--|--+--|-| H |--|   | |
  | |     |   +--------------+  |     | +---+  |   | |
  | |     |                     |     |        |   | |
  | |  B  |   +------ D -----+  |     |        | J |-|--+
  | |     |   | +---+  +---+ |  |     |        |   | |
  | |     |   | |   |--|   | |  |     | +---+  |   | |
  | |     |---|-| E |  | F |-|--|--+--|-| I |--|   | |
  | |     |   | |   |--|   | |  |     | +---+  +---+ |
  | |     |   | +---+  +---+ |  |     +--------------+
  | +-----+   +--------------+  |
  +-----------------------------+


void A::declareProcessOrder() {
  declareProcessStep(ChainFrom(B));
}

void D::declareProcessOrder() {
  declareProcessStep(ChainFrom(E));
}

void G::declareProcessOrder() {
  declareProcessStep(ChainFrom(H));
  declareProcessStep(ChainFrom(I));
}


Should return:

   - C -
  /     \
 B       H - J - I - J
  \     /
   E - F

or the very smartass way:

   - C - H -
  /         \
 B           J
  \         /
   E - F - I



 */
