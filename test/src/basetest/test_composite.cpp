#include "essentia_gtest.h"
#include "../../../src/scheduler/network.h"
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

  while (vinput->process() == OK) ;
  while (composite->process() == OK);
  while (voutput->process() == OK);

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
