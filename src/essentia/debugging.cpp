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
  case EScheduler:  return "[Scheduler ] ";

  case EPython:     return "[  PYTHON  ] ";
  case EPyBindings: return "[  PYBIND  ] ";
  case EUnittest:   return "[ UNITTEST ] ";

  case EUser1:      return "[  USER1   ] ";
  case EUser2:      return "[  USER2   ] ";

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


DebuggingScheduleVector _schedule;

int _savedDebugLevels = ENone;

void scheduleDebug(const DebuggingScheduleVector& schedule) {
  _schedule = schedule;
}

void scheduleDebug(DebuggingSchedule schedule, int nentries) {
  _schedule.resize(nentries);
  for (int i=0; i<nentries; i++) {
    _schedule[i].first.first = schedule[i][0];
    _schedule[i].first.second = schedule[i][1];
    _schedule[i].second = schedule[i][2];
  }
}

void restoreDebugLevels() {
   activatedDebugLevels = _savedDebugLevels;
}

void saveDebugLevels() {
  _savedDebugLevels = activatedDebugLevels;
}

void setDebugLevelForTimeIndex(int index) {
  restoreDebugLevels();
  for (int i=0; i<(int)_schedule.size(); i++) {
    if (_schedule[i].first.first <= index && index <= _schedule[i].first.second) {
      setDebugLevel(_schedule[i].second);
    }
  }
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
