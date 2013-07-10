
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
