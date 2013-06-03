#include "essentia_gtest.h"
#include "network.h"
#include "vectorinput.h"
#include "vectoroutput.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;


TEST(VectorInput, NotConnected) {
  vector<int> output;
  int array[] = {1, 2, 3, 4};
  VectorInput<int>* gen = new VectorInput<int>(array);
  //gen->output("data")  >>  output;

  ASSERT_THROW(scheduler::Network(gen).run(), EssentiaException);
}
