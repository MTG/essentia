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
#include "scheduler/network.h"
#include "streamingalgorithmcomposite.h"
#include "vectorinput.h"
#include "vectoroutput.h"
#include "customalgos.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;




TEST(Composite, SimplePush) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  float input[] = { 1, 2, 3 };
  vector<float> output;

  Algorithm* vinput = new VectorInput<float>(input);
  Algorithm* composite = factory.create("CompositeAlgo");
  Algorithm* voutput = new VectorOutput<float>(&output);

  DBG("connecting");
  connect(vinput->output("data"), composite->input("src"));
  connect(composite->output("dest"), voutput->input("data"));

  DBG("running");
  vinput->process();

  EXPECT_EQ(vinput->output("data").totalProduced(), 1);
  EXPECT_EQ(composite->input("src").available(), 1);

  composite->process();
  voutput->process();

  EXPECT_EQ(output[0], 1);

  while (vinput->process() == OK) ;
  while (composite->process() == OK);
  while (voutput->process() == OK);

  vector<float> expected = arrayToVector<float>(input);
  EXPECT_VEC_EQ(output, expected);

  delete vinput; delete composite; delete voutput;
}

TEST(Composite, ReplaceInnerAlgo) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  float input[] = { 1, 2, 3 };
  vector<float> output;

  Algorithm* vinput = new VectorInput<float>(input);
  Algorithm* composite = factory.create("CompositeAlgo");
  Algorithm* voutput = new VectorOutput<float>(&output);

  connect(vinput->output("data"), composite->input("src"));
  connect(composite->output("dest"), voutput->input("data"));

  vinput->process();
  composite->process();
  voutput->process();

  EXPECT_EQ(output[0], 1);

  while (vinput->process() == OK) ;
  while (composite->process() == OK);
  while (voutput->process() == OK);

  vector<float> expected = arrayToVector<float>(input);
  EXPECT_VEC_EQ(output, expected);

  // reset the network
  DBG("reset network");
  vinput->reset();
  DBG("reset network");
  composite->reset();
  DBG("reset network");
  voutput->reset();
  output.clear();

  // rerun the network
  DBG("rerun network");
  vinput->process();

  DBG("rerun network1");
  EXPECT_EQ(vinput->output("data").totalProduced(), 1);
  EXPECT_EQ(composite->input("src").available(), 1);

  DBG("rerun network2");
  composite->process();
  voutput->process();
  DBG("rerun network3");

  EXPECT_EQ(output[0], 1);

  while (vinput->process() == OK)
      ;
  while (composite->process() == OK)
      ;
  while (voutput->process() == OK)
      ;

  EXPECT_VEC_EQ(output, expected);

  delete vinput; delete composite; delete voutput;
}



TEST(Composite, ExecutionNetwork) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  float input[] = { 1, 2, 3 };
  vector<float> output;

  Algorithm* vinput = new VectorInput<float>(input);
  Algorithm* composite = factory.create("CompositeAlgo");
  Algorithm* voutput = new VectorOutput<float>(&output);

  connect(vinput->output("data"), composite->input("src"));
  connect(composite->output("dest"), voutput->input("data"));

  Network(vinput, false);

  delete vinput; delete composite; delete voutput;
}
