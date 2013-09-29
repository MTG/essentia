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

#include "essentiautil.h"

#ifdef OS_WIN32
#include <fcntl.h>
#include <io.h> // _mktemp
#endif // OS_WIN32

using namespace std;

namespace essentia {

#ifdef OS_WIN32
#define _S_IREAD 256
#define _S_IWRITE 128
int mkstemp(char *tmpl) {
  int ret=-1;

  _mktemp(tmpl);
  ret=open(tmpl,O_RDWR|O_BINARY|O_CREAT|O_EXCL|_O_SHORT_LIVED, _S_IREAD|_S_IWRITE);
  return ret;
}
#endif // OS_WIN32

} //namespace essentia
