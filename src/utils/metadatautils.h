
/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_METADATAUTILS_H
#define ESSENTIA_METADATAUTILS_H

#include <memory>
#include "algorithmfactory.h"
#include "vectoroutput.h"
#include "source.h"


namespace essentia {

void pcmMetadata(const std::string& filename, int& sr, int& ch, int& bitrate) {

  std::string ext = filename.substr(filename.rfind('.'), std::string::npos);
  if (ext != ".wav" && ext != ".aiff" && ext != ".aif") {
    throw EssentiaException("metadatautils: pcmMetadata cannot read files which are neither \"wav\" nor \"aiff");
  }

  // (trick) create an audioloader to know the original samplerate
  std::auto_ptr<streaming::Algorithm> audioloader(streaming::AlgorithmFactory::create("AudioLoader",
                                                                                      "filename", filename));

  sr = (int)streaming::lastTokenProduced<Real>(audioloader->output("sampleRate"));
  ch = streaming::lastTokenProduced<int>(audioloader->output("numberChannels"));
  bitrate = int(16.0/1000.0*sr*ch);
}

} // namespace essentia

#endif
