/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "fileoutputproxy.h"
#include "fileoutput.h"
#include <complex>
#include "tnt/tnt.h"
#include "streamutil.h"

using namespace std;

#define CREATE_FILE_STORAGE(type, proxy)                             \
  if (sameType(sourceType, typeid(type))) {                          \
    fs = new FileOutput<type>();                                     \
    fs->configure("filename", proxy.parameter("filename").toString(),\
                  "mode", proxy.parameter("mode").toString());       \
  }

namespace essentia {
namespace streaming {

const char* FileOutputProxy::name = "FileOutput";
const char* FileOutputProxy::description = DOC("Stores alphanumeric data into text or binary files");

void connect(SourceBase& source, FileOutputProxy& file) {
  const type_info& sourceType = source.typeInfo();

  Algorithm* fs = 0;
  CREATE_FILE_STORAGE(int, file);
  CREATE_FILE_STORAGE(Real, file);
  CREATE_FILE_STORAGE(vector<Real>, file);
  CREATE_FILE_STORAGE(complex<Real>, file);
  CREATE_FILE_STORAGE(vector<complex<Real> >, file);
  CREATE_FILE_STORAGE(std::string, file);
  CREATE_FILE_STORAGE(vector<std::string>, file);
  CREATE_FILE_STORAGE(TNT::Array1D<Real>, file);
  CREATE_FILE_STORAGE(TNT::Array2D<Real>, file);

  if (!fs) throw EssentiaException("FileOutputProxy: File Storage doesn't work for type: ", nameOfType(sourceType));
  else file.setFileStorage(fs);

  connect(source, fs->input("data"));
}

void connect(SourceBase& source, Algorithm& file) {
  FileOutputProxy* proxy = dynamic_cast<FileOutputProxy*>(&file);
  if (!proxy) throw EssentiaException("Cannot connect source ", source.fullName(), " to algorithm ", file.name());
  connect(source, *proxy);
}

} // namespace streaming
} // namespace essentia
