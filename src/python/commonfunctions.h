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

#ifndef COMMONFUNCTIONS_H
#define COMMONFUNCTIONS_H

#include "essentiamath.h"
#include "essentiautil.h"
#include "parsing.h"

void printFixWidth(const std::string& indent, const int& max_width,
                   const std::string& text, ostringstream& result) {
  int printableWidth = max_width-int(indent.size());

  // divide the text into lines
  int pos = 0;
  while (pos < int(text.size())) {
    string candidateStr = text.substr(pos, printableWidth);

    //check for newlines
    size_t newlinePos = candidateStr.find("\n");

    if (newlinePos != string::npos) {
      // contains a newline
      candidateStr = candidateStr.substr(0, int(newlinePos)+1);
      result << indent << candidateStr;
      pos += int(newlinePos)+1;
      continue;
    }

    // if small enough, just output and do newline
    if (int(candidateStr.size()) < printableWidth) {
      result << indent << candidateStr << "\n";
      break;
    }

    // don't create newline in the middle of a word
    size_t lastspacePos = candidateStr.rfind(" ");
    if (lastspacePos == string::npos) {
      // doesn't contain space, can't find a place to break
      result << indent << candidateStr << "\n";
      pos += printableWidth;
      continue;
    }

    // contains a space
    string temp = candidateStr.substr(0, int(lastspacePos));

    // check for only spaces
    bool onlySpaces = true;
    for (int i=0; i<int(temp.size()); ++i) if (temp[i] != ' ') {onlySpaces = false; break;}

    if (onlySpaces) {
      result << indent << candidateStr << "\n";
      pos += int(candidateStr.size());
    }
    else {
      result << indent << temp << "\n";
      pos += int(lastspacePos)+1;
    }
  }
}

template <typename T>
std::string generateDocString(T& algo, const std::string& description) {
  ostringstream docStr;

  docStr << algo.name() << "\n";

  // inputs
  vector<string> inputNames = algo.inputNames();

  if (!inputNames.empty()) {
    docStr << "\n\nInputs:\n";

    // first pass, find largest typename
    int largestTypename = 0;
    for (int i=0; i<int(inputNames.size()); ++i) {
      int typenameLen = int(edtToString( typeInfoToEdt( algo.input(inputNames[i]).typeInfo() ) ).size());
      if (typenameLen > largestTypename) largestTypename = typenameLen;
    }

    // second pass, actually output stuff
    for (int i=0; i<int(inputNames.size()); ++i) {
      string tpname = edtToString( typeInfoToEdt( algo.input(inputNames[i]).typeInfo() ) );
      docStr << "\n  ";
      int nExtraSpace = largestTypename - int(tpname.size());
      for (int j=0; j<nExtraSpace; ++j) docStr << " ";
      docStr << "[" << toLower(tpname) << "] ";
      docStr << inputNames[i] << " - " << algo.inputDescription[inputNames[i]];

    }
    docStr << "\n";
  }

  // outputs
  vector<string> outputNames = algo.outputNames();

  if (!outputNames.empty()) {
    docStr << "\n\nOutputs:\n";

    // first pass, find largest typename
    int largestTypename = 0;
    for (int i=0; i<int(outputNames.size()); ++i) {
      int typenameLen = int(edtToString( typeInfoToEdt( algo.output(outputNames[i]).typeInfo() ) ).size());
      if (typenameLen > largestTypename) largestTypename = typenameLen;
    }

    // second pass, actually output stuff
    for (int i=0; i<int(outputNames.size()); ++i) {
      string tpname = edtToString( typeInfoToEdt( algo.output(outputNames[i]).typeInfo() ) );
      docStr << "\n  ";
      int nExtraSpace = largestTypename - int(tpname.size());
      for (int j=0; j<nExtraSpace; ++j) docStr << " ";
      docStr << "[" << toLower(tpname) << "] ";
      docStr << outputNames[i] << " - " << algo.outputDescription[outputNames[i]];
    }
    docStr << "\n";
  }

  // parameters
  ParameterMap pm = algo.defaultParameters();

  if (!pm.empty()) {
    docStr << "\n\nParameters:\n";
    for (ParameterMap::const_iterator i = pm.begin(); i != pm.end(); ++i) {
      docStr << "\n  " << i->first << ":\n";
      docStr << "    " << toLower(edtToString(paramTypeToEdt(algo.parameter(i->first).type())));
      if (!algo.parameterRange[i->first].empty()) {
        docStr << " ∈ " << algo.parameterRange[i->first];
      }
      if (algo.parameter(i->first).isConfigured()) {
        docStr << " (default = " << algo.parameter(i->first) << ")";
      }
      docStr << "\n";
      printFixWidth("    ", 80, algo.parameterDescription[i->first], docStr);
    }
  }

  // description
  docStr << "\n\nDescription:\n\n";
  printFixWidth("  ", 80, description, docStr);

  return docStr.str();
}


inline PyObject* fromString(const string& str) {
  return PyString_FromStringAndSize(str.c_str(), str.size());
}

template <typename T>
PyObject* generateDocStruct(T& algo, const std::string& description) {
  PyObject* result = PyDict_New();
  PyDict_SetItemString(result, "name", fromString(algo.name()));

  // inputs
  PyObject* inputs = PyList_New(0);
  vector<string> inputNames = algo.inputNames();

  for (int i=0; i<(int)inputNames.size(); i++) {
    PyObject* input = PyDict_New();
    string name = inputNames[i];
    PyDict_SetItemString(input, "name", fromString(name));
    PyDict_SetItemString(input, "type", fromString(toLower(edtToString(typeInfoToEdt(algo.input(name).typeInfo())))));
    PyDict_SetItemString(input, "description", fromString(algo.inputDescription[name]));

    PyList_Append(inputs, input);
  }

  PyDict_SetItemString(result, "inputs", inputs);

  // outputs
  PyObject* outputs = PyList_New(0);
  vector<string> outputNames = algo.outputNames();

  for (int i=0; i<(int)outputNames.size(); i++) {
    PyObject* output = PyDict_New();
    string name = outputNames[i];
    PyDict_SetItemString(output, "name", fromString(name));
    PyDict_SetItemString(output, "type", fromString(toLower(edtToString(typeInfoToEdt(algo.output(name).typeInfo())))));
    PyDict_SetItemString(output, "description", fromString(algo.outputDescription[name]));

    PyList_Append(outputs, output);
  }

  PyDict_SetItemString(result, "outputs", outputs);

  // parameters
  PyObject* params = PyList_New(0);
  ParameterMap pm = algo.defaultParameters();

  for (ParameterMap::const_iterator it = pm.begin(); it != pm.end(); ++it) {
    PyObject* param = PyDict_New();
    string name = it->first;
    PyDict_SetItemString(param, "name", fromString(name));
    PyDict_SetItemString(param, "description", fromString(algo.parameterDescription[name]));
    PyDict_SetItemString(param, "type", fromString(toLower(edtToString(paramTypeToEdt(algo.parameter(name).type())))));

    if (!algo.parameterRange[name].empty()) {
      PyDict_SetItemString(param, "range", fromString(algo.parameterRange[name]));
    }
    else {
      Py_INCREF(Py_None);
      PyDict_SetItemString(param, "range", Py_None);
    }

    if (algo.parameter(name).isConfigured()) {
      PyDict_SetItemString(param, "default", fromString((Stringifier() << algo.parameter(name).toString(6)).str()));
    }
    else {
      Py_INCREF(Py_None);
      PyDict_SetItemString(param, "default", Py_None);
    }

    PyList_Append(params, param);
  }

  PyDict_SetItemString(result, "parameters", params);

  // description
  PyDict_SetItemString(result, "description", fromString(description));

  return result;
}

#endif
