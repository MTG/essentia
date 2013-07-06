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
#include "graphutils.h"
#include "networkparser.h"
#include "customalgos.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


string getName(NetworkNode* node) {
  return removeNodeIdFromName(node->algorithm()->name());
}


TEST(TreeTraversal, DepthFirstMap) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* loader = factory.create("MonoLoader",
                                     "filename", "test/audio/recorded/cat_purrrr.wav");

  //Algorithm* tonal = factory.create("TonalExtractor");
  Algorithm* lowpass = factory.create("LowPass");

  Pool pool;

  connect(loader->output("audio"), lowpass->input("signal"));
  //connect(lowpass->output("signal"), pool, "lp_audio");
  connect(lowpass->output("signal"), NOWHERE);

  Network network(loader);

  vector<string> names = depthFirstMap(network.visibleNetworkRoot(), getName);

  const char* expected[] = { "MonoLoader", "LowPass", "DevNull" };

  EXPECT_VEC_EQ(arrayToVector<string>(expected), names);
}

TEST(TreeTraversal, DepthFirstSearch) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* loader = factory.create("MonoLoader",
                                     "filename", "test/audio/recorded/cat_purrrr.wav");
  Algorithm* lowpass = factory.create("LowPass");
  Pool pool;

  connect(loader->output("audio"), lowpass->input("signal"));
  connect(lowpass->output("signal"), NOWHERE);

  Network network(loader);

  vector<NetworkNode*> dfs = depthFirstSearch(network.visibleNetworkRoot());

  const char* expected[] = { "MonoLoader", "LowPass", "DevNull" };
  int size = ARRAY_SIZE(expected);

  ASSERT_EQ(dfs.size(), (size_t)size);
  for (int i=0; i<size; i++) {
    EXPECT_EQ(removeNodeIdFromName(dfs[i]->algorithm()->name()), expected[i]);
  }
}


TEST(TreeTraversal, NetworkTopologyComparison) {
#define NEW_NODE(name, type) NetworkNode* name = new NetworkNode(AlgorithmFactory::create(#type))
  NEW_NODE(nodeA, A);
  NEW_NODE(nodeA2, A);

  ASSERT_TRUE(areNetworkTopologiesEqual(nodeA, nodeA2));

  //NetworkNode* nodeB = new NetworkNode(AlgorithmFactory::create("B"));
  NEW_NODE(nodeB, B);

  ASSERT_FALSE(areNetworkTopologiesEqual(nodeA, nodeB));

  //            B               B        //
  //           / \             /         //
  //  Compare A   C    With   A          //
  //           \ /             \         //
  //            B               B - C    //

  NEW_NODE(nodeB_, B);
  NEW_NODE(nodeC, C);

  NEW_NODE(nodeB2, B);
  NEW_NODE(nodeB2_, B);
  NEW_NODE(nodeC2, C);

  nodeA->addChild(nodeB);
  nodeA->addChild(nodeB_);
  nodeB->addChild(nodeC);
  nodeB_->addChild(nodeC);

  nodeA2->addChild(nodeB2);
  nodeA2->addChild(nodeB2_);
  nodeB2_->addChild(nodeC2);

  ASSERT_FALSE(areNetworkTopologiesEqual(nodeA, nodeA2));

  nodeB2->addChild(nodeC2);
  ASSERT_TRUE(areNetworkTopologiesEqual(nodeA, nodeA2));

#define DELETE_NODE(node) delete node->algorithm(); delete node

  DELETE_NODE(nodeA); DELETE_NODE(nodeA2); DELETE_NODE(nodeB); DELETE_NODE(nodeB_);
  DELETE_NODE(nodeC); DELETE_NODE(nodeB2); DELETE_NODE(nodeB2_); DELETE_NODE(nodeC2);
}

TEST(TreeTraversal, NetworkTopologyComparisonWithParser) {
  const char* network1[] = {
    "            +---+               ",
    "         +--| B |--+            ",
    "         |  +---+  |            ",
    "   +---+ |         |  +---+     ",
    "   | A |-+         +--| C |     ",
    "   +---+ |         |  +---+     ",
    "         |  +---+  |            ",
    "         +--| B |--+            ",
    "            +---+               "
  };


  const char* network2[] = {
    "           +---+                ",
    "         +-| B |                ",
    "         | +---+                ",
    "   +---+ |                      ",
    "   | A |-+                      ",
    "   +---+ |                      ",
    "         | +---+      +---+     ",
    "         +-| B |------| C |     ",
    "           +---+      +---+     "
  };

  DBG("---------------------------------------------");
  DBG("parsing NP1");

  NetworkParser np1(network1, false);
  DBG("parsing NP2");
  NetworkParser np2(network2, false);

  DBG("comparing");
  ASSERT_FALSE(areNetworkTopologiesEqual(np1.network()->visibleNetworkRoot(),
                                         np2.network()->visibleNetworkRoot()));
}


TEST(TreeTraversal, NetworkTopologyComparisonWithParser2) {
  const char* network1[] = {
    "           +---+      +---+     ",
    "         +-| B |------| C |     ",
    "         | +---+      +---+     ",
    "   +---+ |                      ",
    "   | A |-+                      ",
    "   +---+ |                      ",
    "         | +---+                ",
    "         +-| B |                ",
    "           +---+                "
  };

  const char* network2[] = {
    "           +---+              ",
    "         +-| B |              ",
    "         | +---+              ",
    "   +---+ |                    ",
    "   | A |-+                    ",
    "   +---+ |                    ",
    "         | +---+    +---+     ",
    "         +-| B |----| C |     ",
    "           +---+    +---+     "
  };

  NetworkParser np1(network1, false);
  NetworkParser np2(network2, false);

  ASSERT_TRUE(areNetworkTopologiesEqual(np1.network()->visibleNetworkRoot(),
                                        np2.network()->visibleNetworkRoot()));
}

TEST(TreeTraversal, NetworkTopologyComparisonComplex) {
  const char* network1[] = {
    "                                                                               ",
    "                                                                               ",
    "               +------+                                                        ",
    "            +--|  B   |-+                                                      ",
    "            |  +------+ |                                                      ",
    "    +-----+ |           |                                                      ",
    "    |  A  |-+           |  +-------+         +------+                          ",
    "    +-----+ |           ++-|   C   |---------|  D   |                          ",
    "            |            | +-------+         |      |----+                     ",
    "            | +-----+    |                   +------+    |  +--------+         ",
    "            +-|  B  |----+                               +--|   E    |         ",
    "              |     |--+                              +-----|        |         ",
    "              +-----+  |      +-------+               |     +--------+         ",
    "                       +------|  D    |               |                        ",
    "                              |       |---------------+                        ",
    "                              +-------+                                        ",
    "                                                                               "
  };

  const char* network2[] = {
    "                                              +----------+                     ",
    "                            +-------+   +-----|   D      |--+                  ",
    "                        +---|   B   |---+     +----------+  |                  ",
    "            +-----+     |   |       |----+                  |                  ",
    "            |  A  |--+  |   +-------+    |                  |                  ",
    "            +-----+  +--+                |                  |                  ",
    "                     |           +-------+                  |                  ",
    "   +-----------------+           |                          |                  ",
    "   |                             |                          |                  ",
    "   |  +-------+                  |   +--------+             |                  ",
    "   +--|   B   |------------------+---|   C    |---+         |                  ",
    "      +-------+                      +--------+   |         |                  ",
    "                                                  |         |  +-------+       ",
    "       +---------------------+                    |         +--|   E   |       ",
    "       |   +----------+      +--------------------+         |  +-------+       ",
    "       +---|    D     |                                     |                  ",
    "           |          |-------------------------------------+                  ",
    "           +----------+                                                        ",
    "                                                                               "
  };

  NetworkParser np1(network1, false);
  NetworkParser np2(network2, false);

  ASSERT_TRUE(areNetworkTopologiesEqual(np1.network()->visibleNetworkRoot(),
                                        np2.network()->visibleNetworkRoot()));
}
