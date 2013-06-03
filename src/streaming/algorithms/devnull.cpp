#include "devnull.h"
#include "tnt/tnt.h"
using namespace std;

#define CREATE_DEVNULL(type) if (sameType(sourceType, typeid(type))) devnull = new DevNull<type>();

namespace essentia {
namespace streaming {


void connect(SourceBase& source, DevNullConnector dummy) {
  const type_info& sourceType = source.typeInfo();

  Algorithm* devnull = 0;
  CREATE_DEVNULL(int);
  CREATE_DEVNULL(Real);
  CREATE_DEVNULL(vector<Real>);
  CREATE_DEVNULL(string);
  CREATE_DEVNULL(vector<string>);
  CREATE_DEVNULL(TNT::Array2D<Real>);
  CREATE_DEVNULL(StereoSample);

  if (!devnull) throw EssentiaException("DevNull class doesn't work for type: ", nameOfType(sourceType));

  connect(source, devnull->input("data"));
}


void disconnect(SourceBase& source, DevNullConnector devnull) {
  // find dev null that this source is connected to, and disconnect it
  // note its ok we modify the list we're interating over since we return as
  // soon as we make a modification
  for (int i=0; i<int(source.sinks().size()); ++i) {
    SinkBase& sink = *(source.sinks()[i]);
    Algorithm* sinkAlg = sink.parent();

    // TODO: huh?
    if (sinkAlg->name() == "DevNull") {
      disconnect(source, sink);

      // since the DevNull is no longer connected to a network, it must be
      // manually deleted (no one should have a pointer to this instance)
      delete sinkAlg;
      return;
    }
  }

  ostringstream msg;
  msg << "the source you are disconnecting (";
  msg << source.fullName();
  msg << ") is not connected to NOWHERE";
  throw EssentiaException(msg);
}


} // namespace streaming
} // namespace essentia
