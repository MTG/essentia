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

#ifndef ESSENTIA_DEBUGGING_H
#define ESSENTIA_DEBUGGING_H

#include <deque>
#include <string>
#include "config.h"
#include "streamutil.h"
#include "stringutil.h"

namespace essentia {

// IMPORTANT:
// Make sure that each time you change something in this enum, you reflect the
// same changes in the python bindings, located in src/python/essentia/__init__.py
enum DebuggingModule {

  EAlgorithm   = 1 << 0,
  EConnectors  = 1 << 1,
  EFactory     = 1 << 2,
  ENetwork     = 1 << 3,
  EGraph       = 1 << 4,
  EExecution   = 1 << 5,
  EMemory      = 1 << 6,  // for mem operations, such as new/delete
  EScheduler   = 1 << 7,

  EPython      = 1 << 20, // for use in python scripts
  EPyBindings  = 1 << 21, // for use in python/C module
  EUnittest    = 1 << 22,

  EUser1       = 1 << 25, // freely available for the user
  EUser2       = 1 << 26, // freely available for the user

  ENone        = 0,
  EAll         = (1 << 30) - 1
};

const char* debugModuleDescription(DebuggingModule module);

/**
 * A bitmask representing which debug levels are currently activated.
 */
extern int activatedDebugLevels;

extern bool infoLevelActive;
extern bool warningLevelActive;
extern bool errorLevelActive;

/**
 * An integer representing the indentation with which to print the debug messages
 */
extern int debugIndentLevel;

void setDebugLevel(int levels);
void unsetDebugLevel(int levels);

/**
 * Asynchronous thread-safe logger object. (TODO: implementation is still not thread-safe)
 */
class Logger {
 protected:
  std::deque<std::string> _msgQueue;
  bool _addHeader;

  void flush();

 public:
  Logger() : _addHeader(true) {}

  void debug(DebuggingModule module, const std::string& msg, bool resetHeader = false);
  void info(const std::string& msg);
  void warning(const std::string& msg);
  void error(const std::string& msg);

};

extern Logger loggerInstance;

} // namespace essentia


#if DEBUGGING_ENABLED

#  define E_DEBUG_INDENT debugIndentLevel++
#  define E_DEBUG_OUTDENT debugIndentLevel--

#  define E_ACTIVE(module) ((module) & activatedDebugLevels)
#  define E_STRINGIFY(msg) (Stringifier() << msg).str()

// the if(E_ACTIVE) is an optimization, it is not necessary but avoids sending
// everything to the Stringifier if the debug level is not activated
#  define E_DEBUG_NONL(module, msg) if (E_ACTIVE(module)) loggerInstance.debug(module, E_STRINGIFY(msg), false)
#  define E_DEBUG(module, msg) if (E_ACTIVE(module)) loggerInstance.debug(module, E_STRINGIFY(msg << '\n'), true)

// NB: the following #define macros only work when used inside one of streaming::Algorithm's methods
#  define ALGONAME _name << std::string(std::max(15-(int)_name.size(), 0), ' ') << ": "
#  define EXEC_DEBUG(msg) E_DEBUG(EExecution, ALGONAME << nProcess << " - " << msg)

#  define E_INFO(msg) loggerInstance.info(E_STRINGIFY(msg))
#  define E_WARNING(msg) loggerInstance.warning(E_STRINGIFY(msg))
#  define E_ERROR(msg) loggerInstance.error(E_STRINGIFY(msg))

#else // DEBUGGING_ENABLED

#  define E_DEBUG_INDENT
#  define E_DEBUG_OUTDENT
#  define E_ACTIVE(module) false
#  define E_STRINGIFY(msg) ""
#  define E_DEBUG_NONL(module, msg)
#  define E_DEBUG(module, msg)
#  define ALGONAME
#  define EXEC_DEBUG(msg)
#  define E_INFO(msg)
#  define E_WARNING(msg)
#  define E_ERROR(msg)

#endif // DEBUGGING_ENABLED



#endif // ESSENTIA_DEBUGGING_H
