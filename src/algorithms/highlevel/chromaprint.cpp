/*
 * Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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

#include "chromaprint.h"

using namespace std;
namespace essentia {
namespace standard {

const char* Chromaprint::name = "Chromaprint";
const char* Chromaprint::category = "Fingerprinting";
const char* Chromaprint::description = DOC("");


void Chromaprint::compute(){
  const vector<Real>& signal = _signal.get();
  string& fingerprint = _fingerprint.get();

  const vector<int16_t> signalCast(signal.begin(), signal.end());
  //TODO multiply to 2**16 to use all the int dynamic range.

  ChromaprintContext *ctx;
  char *fp;

  const int sample_rate = 44100;
  const int num_channels = 2;

  ctx = chromaprint_new(CHROMAPRINT_ALGORITHM_DEFAULT);

  chromaprint_start(ctx, sample_rate, num_channels);

  for (unsigned i = 0;i < signal.size(); i++) {
    chromaprint_feed(ctx, &signalCast[i], signal.size());
  }
  //while (it < signalCast.end()) {
  //  chromaprint_feed(ctx, it, signal.size());
  //}
  chromaprint_finish(ctx);

  chromaprint_get_fingerprint(ctx, &fp);

  fingerprint = const_cast<char*>(fp);

  chromaprint_dealloc(fp);

  chromaprint_free(ctx);

/*  @endcode

  Note that there is no error handling in the code above. Almost any of the called functions can fail.
  You should check the return values in an actual code.
 */
}

} // namespace standard
} // namespace essentia
