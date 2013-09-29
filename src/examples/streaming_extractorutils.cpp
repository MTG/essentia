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

#include "streaming_extractorutils.h"
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

void readMetadata(const string& audioFilename, Pool& pool) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* metadata = factory.create("MetadataReader",
                                       "filename", audioFilename,
                                       "failOnError", true);

  connect(metadata->output("title"),    pool, "metadata.tags.title");
  connect(metadata->output("artist"),   pool, "metadata.tags.artist");
  connect(metadata->output("album"),    pool, "metadata.tags.album");
  connect(metadata->output("comment"),  pool, "metadata.tags.comment");
  connect(metadata->output("genre"),    pool, "metadata.tags.genre");
  connect(metadata->output("track"),    pool, "metadata.tags.track");
  connect(metadata->output("year"),     pool, "metadata.tags.year");
  connect(metadata->output("length"),   NOWHERE); // let audio loader take care of this
  connect(metadata->output("bitrate"),  pool, "metadata.audio_properties.bitrate");
  connect(metadata->output("sampleRate"), NOWHERE); // let the audio loader take care of this
  connect(metadata->output("channels"), pool, "metadata.audio_properties.channels");

  Network(metadata).run();
}

void setDefaultOptions(Pool& pool) {
  // general
  pool.set("equalLoudness", true);
  pool.set("nequalLoudness", false);
  pool.set("shortSound", false);
  pool.set("startTime", 0);
  pool.set("endTime", 2000.0);
  pool.set("analysisSampleRate", 44100.0);
  pool.set("outputJSON", false); // false for YAML, true for JSON
  pool.set("outputFrames", false); // set to true to output all frames
  string silentFrames = "noise";
  int zeroPadding = 0;
  string windowType = "hann";
  pool.set("svm.compute", false);
  int size1=1000, inc1=300, size2=600, inc2=50, cpw=5, minlength=10;
  pool.set("segmentation.compute", false);
  pool.set("segmentation.size1", size1);
  pool.set("segmentation.inc1", inc1);
  pool.set("segmentation.size2", size2);
  pool.set("segmentation.inc2", inc2);
  pool.set("segmentation.cpw", cpw );
  pool.set("segmentation.minimumSegmentsLength", minlength);

  // lowlevel
  pool.set("lowlevel.compute", true);
  pool.set("lowlevel.frameSize", 2048);
  pool.set("lowlevel.hopSize", 1024);
  pool.set("lowlevel.zeroPadding", zeroPadding);
  pool.set("lowlevel.windowType", "blackmanharris62");
  pool.set("lowlevel.silentFrames", silentFrames);

  // beattrack
  pool.set("beattrack.compute", true);
  // TODO: Add beattrack parameters

  // average_loudness
  pool.set("average_loudness.compute", true);
  pool.set("average_loudness.frameSize", 88200);
  pool.set("average_loudness.hopSize", 44100);
  pool.set("average_loudness.windowType", windowType);
  pool.set("average_loudness.silentFrames", silentFrames);

  // rhythm
  pool.set("rhythm.compute", true);
  //pool.set("rhythm.frameSize", 1024);
  //pool.set("rhythm.hopSize", 256);
  //pool.set("rhythm.useOnset", true);
  //pool.set("rhythm.useBands", true);
  //pool.set("rhythm.numberFrames", 1024);
  //pool.set("rhythm.frameHop", 1024);
  pool.set("rhythm.method", "degara");
  pool.set("rhythm.minTempo", 40);
  pool.set("rhythm.maxTempo", 208);

  // tonal
  pool.set("tonal.compute", true);
  pool.set("tonal.frameSize", 4096);
  pool.set("tonal.hopSize", 2048);
  pool.set("tonal.zeroPadding", zeroPadding);
  pool.set("tonal.windowType", "blackmanharris62");
  pool.set("tonal.silentFrames", silentFrames);

  // sfx
  pool.set("sfx.compute", true);

  // highlevel
  pool.set("highlevel.compute", true);

  // panning
  pool.set("panning.compute", false);
  pool.set("panning.frameSize", 8192);
  pool.set("panning.hopSize", 2048);
  pool.set("panning.zeroPadding", 8192);
  pool.set("panning.windowType", "hann");
  pool.set("panning.silentFrames", silentFrames);

  // stats
  const char* statsArray[] = { "mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };
  const char* mfccStatsArray[] = { "mean", "cov", "icov" };
  vector<string> stats = arrayToVector<string>(statsArray);
  vector<string> mfccStats = arrayToVector<string>(mfccStatsArray);
  for (int i=0; i<(int)stats.size(); i++) {
    pool.add("lowlevel.stats", stats[i]);
    pool.add("tonal.stats", stats[i]);
    pool.add("rhythm.stats", stats[i]);
    pool.add("sfx.stats", stats[i]);
  }
  for (int i=0; i<(int)mfccStats.size(); i++)
    pool.add("lowlevel.mfccStats", mfccStats[i]);
  pool.add("panning.stats", "copy");
}

void setOptions(Pool& options, const std::string& filename) {
  setDefaultOptions(options);
  if (filename.empty()) return;
  Pool opts;
  standard::Algorithm * yaml = standard::AlgorithmFactory::create("YamlInput", "filename", filename);
  yaml->output("pool").set(opts);
  yaml->compute();
  delete yaml;
  options.merge(opts, "replace");
  //const vector<string>& descriptorNames = options.descriptorNames();
  //for (int i=0; i<(int)descriptorNames.size(); i++) {
  //  cout << descriptorNames[i] << endl;
  //}
}

void mergeOptionsAndResults(Pool& results, const Pool& options) {
  // merges the configuration results with results pool
  results.set("configuration.general.equalLoudness",        options.value<Real>("equalLoudness"));
  results.set("configuration.general.nequalLoudness",       options.value<Real>("nequalLoudness"));
  results.set("configuration.general.shortSound",           options.value<Real>("shortSound"));
  results.set("configuration.general.startTime",            options.value<Real>("startTime"));
  results.set("configuration.general.endTime",              options.value<Real>("endTime"));
  results.set("configuration.general.analysisSampleRate",   options.value<Real>("analysisSampleRate"));
  results.set("configuration.svm.compute",          options.value<Real>("svm.compute"));
  results.set("configuration.segmentation.compute", options.value<Real>("segmentation.compute"));
  results.set("configuration.segmentation.size1",   options.value<Real>("segmentation.size1"));
  results.set("configuration.segmentation.inc1",    options.value<Real>("segmentation.inc1"));
  results.set("configuration.segmentation.size2",   options.value<Real>("segmentation.size2"));
  results.set("configuration.segmentation.inc2",    options.value<Real>("segmentation.inc2"));
  results.set("configuration.segmentation.cpw",     options.value<Real>("segmentation.cpw"));
  results.set("configuration.segmentation.minimumSegmentsLength",
           options.value<Real>("segmentation.minimumSegmentsLength"));

  // lowlevel
  results.set("configuration.lowlevel.compute",      options.value<Real>("lowlevel.compute"));
  results.set("configuration.lowlevel.frameSize",    options.value<Real>("lowlevel.frameSize"));
  results.set("configuration.lowlevel.hopSize",      options.value<Real>("lowlevel.hopSize"));
  results.set("configuration.lowlevel.zeroPadding",  options.value<Real>("lowlevel.zeroPadding"));
  results.set("configuration.lowlevel.windowType",   options.value<string>("lowlevel.windowType"));
  results.set("configuration.lowlevel.silentFrames", options.value<string>("lowlevel.silentFrames"));

  // average_loudness
  results.set("configuration.average_loudness.compute",      options.value<Real>("average_loudness.compute"));
  results.set("configuration.average_loudness.frameSize",    options.value<Real>("average_loudness.frameSize"));
  results.set("configuration.average_loudness.hopSize",      options.value<Real>("average_loudness.hopSize"));
  results.set("configuration.average_loudness.windowType",   options.value<string>("average_loudness.windowType"));
  results.set("configuration.average_loudness.silentFrames", options.value<string>("average_loudness.silentFrames"));

  // rhythm
  results.set("configuration.rhythm.compute",      options.value<Real>("rhythm.compute"));
  //results.set("configuration.rhythm.frameSize",    options.value<Real>("rhythm.frameSize"));
  //results.set("configuration.rhythm.hopSize",      options.value<Real>("rhythm.hopSize"));
  //results.set("configuration.rhythm.useOnset",     options.value<Real>("rhythm.useOnset"));
  //results.set("configuration.rhythm.useBands",     options.value<Real>("rhythm.useBands"));
  //results.set("configuration.rhythm.numberFrames", options.value<Real>("rhythm.numberFrames"));
  //results.set("configuration.rhythm.frameHop",     options.value<Real>("rhythm.frameHop"));
  results.set("configuration.rhythm.method", options.value<string>("rhythm.method"));
  results.set("configuration.rhythm.minTempo", options.value<Real>("rhythm.minTempo"));
  results.set("configuration.rhythm.maxTempo", options.value<Real>("rhythm.maxTempo"));

  // tonal
  results.set("configuration.tonal.compute",      options.value<Real>("tonal.compute"));
  results.set("configuration.tonal.frameSize",    options.value<Real>("tonal.frameSize"));
  results.set("configuration.tonal.hopSize",      options.value<Real>("tonal.hopSize"));
  results.set("configuration.tonal.zeroPadding",  options.value<Real>("tonal.zeroPadding"));
  results.set("configuration.tonal.windowType",   options.value<string>("tonal.windowType"));
  results.set("configuration.tonal.silentFrames", options.value<string>("tonal.silentFrames"));

  // sfx
  results.set("configuration.sfx.compute", options.value<Real>("sfx.compute"));

  // panning
  results.set("configuration.panning.compute",      options.value<Real>("panning.compute"));
  results.set("configuration.panning.frameSize",    options.value<Real>("panning.frameSize"));
  results.set("configuration.panning.hopSize",      options.value<Real>("panning.hopSize"));
  results.set("configuration.panning.zeroPadding",  options.value<Real>("panning.zeroPadding"));
  results.set("configuration.panning.windowType",   options.value<string>("panning.windowType"));
  results.set("configuration.panning.silentFrames", options.value<string>("panning.silentFrames"));

  // stats
  vector<string> lowlevelStats = options.value<vector<string> >("lowlevel.stats");
  for (int i=0; i<(int)lowlevelStats.size(); i++) results.add("configuration.lowlevel.stats", lowlevelStats[i]);

  vector<string> tonalStats = options.value<vector<string> >("tonal.stats");
  for (int i=0; i<(int)tonalStats.size(); i++) results.add("configuration.tonal.stats", tonalStats[i]);

  vector<string> rhythmStats = options.value<vector<string> >("rhythm.stats");
  for (int i=0; i<(int)rhythmStats.size(); i++) results.add("configuration.rhythm.stats", rhythmStats[i]);

  vector<string> sfxStats = options.value<vector<string> >("sfx.stats");
  for (int i=0; i<(int)sfxStats.size(); i++) results.add("configuration.sfx.stats", sfxStats[i]);

  vector<string> mfccStats = options.value<vector<string> >("lowlevel.mfccStats");
  for (int i=0; i<(int)mfccStats.size(); i++) results.add("configuration.lowlevel.mfccStats", mfccStats[i]);

  vector<string> panningStats = options.value<vector<string> >("panning.stats");
  for (int i=0; i<(int)panningStats.size(); i++) results.add("configuration.panning.stats", panningStats[i]);
}
