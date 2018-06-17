#include "extractor_utils.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;


void setExtractorDefaultOptions(Pool &options) {
  options.set("outputFrames", false);
  options.set("outputFormat", "json");
  options.set("requireMbid", false);
  options.set("indent", 4);

  options.set("highlevel.inputFormat", "json");
}


void setExtractorOptions(const std::string& filename, Pool& options) {
  if (filename.empty()) return;

  Pool opts;
  Algorithm * yaml = AlgorithmFactory::create("YamlInput", "filename", filename);
  yaml->output("pool").set(opts);
  yaml->compute();
  delete yaml;
  options.merge(opts, "replace");
}


void mergeValues(Pool& pool, Pool& options) {
  // NOTE:
  // - no check for if descriptors with the same names as the ones asked to
  //   merge exist already
  // - all descriptors to be merged are expected to be strings
  // TODO implement a method in Pool to detect the type of a descriptor given its name

  string mergeKeyPrefix = "mergeValues";
  vector<string> keys = options.descriptorNames(mergeKeyPrefix);

  for (int i=0; i<(int) keys.size(); ++i) {
    keys[i].replace(0, mergeKeyPrefix.size()+1, "");
    pool.set(keys[i], options.value<string>(mergeKeyPrefix + "." + keys[i]));
  }
}


void outputToFile(Pool& pool, const string& outputFilename, Pool& options) {

  cerr << "Writing results to file " << outputFilename << endl;
  int indent = (int)options.value<Real>("indent");

  string format = options.value<string>("outputFormat");
  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename,
                                               "doubleCheck", true,
                                               "format", format,
                                               "writeVersion", false,
                                               "indent", indent);
  output->input("pool").set(pool);
  output->compute();
  delete output;
}
