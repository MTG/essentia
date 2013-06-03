#include "debugging.h"
#include <iostream>
using namespace std;

namespace essentia {

bool infoLevelActive = true;
bool warningLevelActive = true;
bool errorLevelActive = true;

int activatedDebugLevels = 0;
int debugIndentLevel = 0;

Logger loggerInstance;

const char* debugModuleDescription(DebuggingModule module) {
  switch (module) {
  case EAlgorithm:  return "[Algorithm ] ";
  case EConnectors: return "[Connectors] ";
  case EFactory:    return "[Factory   ] ";
  case ENetwork:    return "[Network   ] ";
  case EGraph:      return "[Graph     ] ";
  case EExecution:  return "[Execution ] ";
  case EMemory:     return "[Memory    ] ";

  case EPython:     return "[  PYTHON  ] ";
  case EUnittest:   return "[ UNITTEST ] ";

  case ENone:       return "[          ] ";
  case EAll:        return "[   ALL    ] ";
  default:          return "[ Mixed    ] ";
  }
}


void setDebugLevel(int levels) {
  activatedDebugLevels |= levels;
}

void unsetDebugLevel(int levels) {
  activatedDebugLevels &= ~levels;
}



// NOTE: in a thread-safe implementation, the msg queue would be thread-safe and
//       the flushing would need to happen in a separate thread
//       This can be achieved using tbb::concurrent_queue

void Logger::flush() {
  while (!_msgQueue.empty()) {
    std::cout << _msgQueue.front() << std::flush;
    _msgQueue.pop_front();
  }
}

void Logger::debug(DebuggingModule module, const string& msg, bool resetHeader) {
  if (module & activatedDebugLevels) {
    if (_addHeader) {
      _msgQueue.push_back(E_STRINGIFY(debugModuleDescription(module)      // module name
                                      + string(debugIndentLevel * 8, ' ') // indentation
                                      + msg));                            // msg
    }
    else {
      _msgQueue.push_back(msg);
    }

    _addHeader = resetHeader;
    flush();
  }
}

void Logger::info(const string& msg) {
  if (!infoLevelActive) return;
  static const string GREEN_FONT = "\x1B[0;32m";
  static const string RESET_FONT = "\x1B[0m";
  _msgQueue.push_back(E_STRINGIFY(GREEN_FONT << "[   INFO   ] " << RESET_FONT << msg << '\n'));
  flush();
}

void Logger::warning(const string& msg) {
  if (!warningLevelActive) return;
  static const string YELLOW_FONT = "\x1B[0;33m";
  static const string RESET_FONT = "\x1B[0m";
  _msgQueue.push_back(E_STRINGIFY(YELLOW_FONT << "[ WARNING  ] " << RESET_FONT << msg << '\n'));
  flush();
}

void Logger::error(const string& msg) {
  if (!errorLevelActive) return;
  static const string RED_FONT = "\x1B[0;31m";
  static const string RESET_FONT = "\x1B[0m";
  _msgQueue.push_back(E_STRINGIFY(RED_FONT << "[  ERROR   ] " << RESET_FONT << msg << '\n'));
  flush();
}

} // namespace essentia
