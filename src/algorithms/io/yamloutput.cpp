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

#include "yamloutput.h"
#include "essentia.h"
#include "output.h" // ../utils/output
#include <fstream>
#include <sstream> // escapeJsonString


using namespace std;
using namespace essentia;
using namespace standard;

const char* YamlOutput::name = "YamlOutput";
const char* YamlOutput::description = DOC("This algorithm emits a YAML or JSON representation of a Pool.\n"
  "\n"
  "Each descriptor key in the Pool is decomposed into different nodes of the YAML (JSON) format by "
  "splitting on the '.' character. For example a Pool that looks like this:\n"
  "\n"
  "    foo.bar.some.thing: [23.1, 65.2, 21.3]\n"
  "\n"
  "will be emitted as:\n"
  "\n"
  "    metadata:\n"
  "        essentia:\n"
  "            version: <version-number>\n"
  "\n"
  "    foo:\n"
  "        bar:\n"
  "            some:\n"
  "                thing: [23.1, 65.2, 21.3]");

// TODO arrange keys in alphabetical order and make sure to add that to the
// dictionary, when implementing this, it should be made general enough to
// add other sorting mechanisms (eg numerically, by size, custom ordering).

void YamlOutput::configure() {
  _filename = parameter("filename").toString();
  _doubleCheck = parameter("doubleCheck").toBool();
  _outputJSON = (parameter("format").toLower() == "json");

  if (_filename == "") throw EssentiaException("please provide a valid filename");
}


// this function splits a string up on the '.' char and returns a vector of
// strings where each element represents a string between dots.
vector<string> split(const string& s) {
  string::size_type dotpos = s.find('.');
  string::size_type prevdotpos = 0;
  vector<string> result;

  if (dotpos != string::npos) {
    result.push_back(s.substr(0, dotpos));
  }
  else {
    result.push_back(s);
    return result;
  }

  if (dotpos+1 == string::npos) {
    return result;
  }

  prevdotpos = dotpos;
  dotpos = s.find('.', prevdotpos+1);

  while (dotpos != string::npos) {
    if (prevdotpos+1 == string::npos) return result;

    result.push_back( s.substr(prevdotpos + 1, dotpos - (prevdotpos + 1)) );
    prevdotpos = dotpos;
    dotpos = s.find('.', prevdotpos+1);
  }

  // add last bit
  result.push_back( s.substr(prevdotpos+1) );

  return result;
}


// this function escapes utf-8 string to be compatible with JSON standard, 
// but it does not handle invalid utf-8 characters. Values in the pool are 
// expected to be correct utf-8 strings, and it is up to the user to provide
// correct utf-8 strings for the names of descriptors in the Pool. This 
// function is not called for Pool descriptor names, but only for string values.
string escapeJsonString(const string& input) {
  ostringstream escaped;
  for (string::const_iterator i = input.begin(); i != input.end(); i++) { 
    switch (*i) {
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      case '\f': escaped << "\\f"; break;
      case '\b': escaped << "\\b"; break;
      case '"': escaped << "\\\""; break;
      case '/': escaped << "\\/"; break;
      case '\\': escaped << "\\\\"; break;
      default: escaped << *i; break;
    } 
  }
  return escaped.str();
}

// A YamlNode represents a node in the YAML tree. A YamlNode without any value
// is valid, it is simply a namespace identifier. It is required that every
// *leaf* node in a YAML tree have a defined value though.
struct YamlNode {
  string name;
  Parameter* value;
  vector<YamlNode*> children;

  YamlNode(const string& n) : name(n), value(0) {}

  ~YamlNode () {
    delete value;
    for (int i=0; i<(int)children.size(); ++i) {
      delete children[i];
    }
  }
};


template <typename IterType>
void fillYamlTreeHelper(YamlNode* root, const IterType it) {
  vector<string> pathparts = split(it->first);
  YamlNode* currNode = root;

  // iterate over each of the pieces of the path
  for (int i=0; i<(int)pathparts.size(); ++i) {
    bool newNode = true;

    // search to see if the path part is already in the tree
    for (int j=0; j<(int)currNode->children.size(); ++j) {
      if (currNode->children[j]->name == pathparts[i]) { // already in the tree
        newNode = false;
        currNode = currNode->children[j];
        break;
      }
    }

    if (newNode) { // path part was not found in the tree
      YamlNode* newNode = new YamlNode(pathparts[i]);
      currNode->children.push_back(newNode);
      currNode = newNode;
    }
  }

  // end of the path
  currNode->value = new Parameter(it->second);
}

/*
 fillYamlTree (Pool, root YamlNode):
 Places all the values from the pool <string, val> into a tree
 e.g. a pool like this:
   foo1.bar  [134.2, 343.234]
   foo2.bar  ["hello"]


 will get translated to a tree like this:

                     __root
                    /      \
                   /        \
                  /          \
                 /            \
                /              \
              foo1             foo2
              /                  \
             /                    \
   bar ->[ 134.2, 343.234]    bar ->[ "hello"]


 And the YAML will look like:

 foo1:
     bar: [ 134.2, 343.234]
 foo2:
     bar: [ "hello"]


*/

void fillYamlTree (const Pool& p, YamlNode* root) {
  #define FILL_YAML_TREE_MACRO(type, tname)                                    \
  for (map<string, type >::const_iterator it = p.get##tname##Pool().begin();   \
       it != p.get##tname##Pool().end(); ++it) {                               \
    fillYamlTreeHelper(root, it);                                              \
  }

  FILL_YAML_TREE_MACRO(Real, SingleReal);
  FILL_YAML_TREE_MACRO(vector<Real>, Real);
  FILL_YAML_TREE_MACRO(vector<Real>, SingleVectorReal);
  FILL_YAML_TREE_MACRO(vector<vector<Real> >, VectorReal);

  FILL_YAML_TREE_MACRO(string, SingleString);
  FILL_YAML_TREE_MACRO(vector<string>, String);
  FILL_YAML_TREE_MACRO(vector<vector<string> >, VectorString);

  FILL_YAML_TREE_MACRO(vector<TNT::Array2D<Real> >, Array2DReal);
  FILL_YAML_TREE_MACRO(vector<StereoSample>, StereoSample);

  #undef FILL_YAML_TREE_MACRO
}


// Emits YAML given a YamlNode root to a specified stream.
// This is a recursive solution.
template <typename StreamType>
void emitYaml(StreamType* s, YamlNode* n, const string& indent) {
  *s << indent << n->name << ":";

  if (n->children.empty()) { // if there are no children, emit the value here
    if (n->value != NULL) {
      *s << " " << *(n->value) << "\n";  // Parameters know how to be emitted to streams
    }
    else { // you should never have this case: a key without any children or associated value
      throw EssentiaException("YamlOutput: input pool is invalid, contains key with no associated value");
    }
  }
  else {
    // we can make the assumption that this node has no value because the pool
    // doesn't not allow parent nodes to have values
    if (n->value != NULL) {
      throw EssentiaException(
          "YamlOutput: input pool is invalid, a parent key should not have a"
          "value in addition to child keys");
    }

    *s << "\n";

    // and then emit the yaml for all of its children, recursive call
    for (int i=0; i<(int)n->children.size(); ++i) {
      emitYaml(s, n->children[i], indent+"    ");
    }
  }
}


template <typename StreamType>
void emitJson(StreamType* s, YamlNode* n, const string& indent) {
  *s << indent << "\"" << n->name << "\":";

  if (n->children.empty()) { // if there are no children, emit the value here
    if (n->value != NULL) {

      // Escape string or vector of strings values for json compatibility
      // FIXME Instead, is it possible to add an option to escape strings inside '<<'
      // implementation for Parameters themselves?
      Parameter::ParamType nodeType = (*(n->value)).type();
      if (nodeType == Parameter::STRING) {
        *s << " " << "\"" << escapeJsonString((*(n->value)).toString()) << "\"";
      }
      else if (nodeType == Parameter::VECTOR_STRING) {
        vector<string> escaped = (*(n->value)).toVectorString();
        for (size_t i=0; i<escaped.size(); ++i) {
          escaped[i] = "\"" + escapeJsonString(escaped[i]) + "\"";
        }
        *s << " " << escaped;
      }
      else {
        *s << " " << *(n->value); // Parameters know how to be emitted to streams
      }
    }
    else { // you should never have this case: a key without any children or associated value
      throw EssentiaException("JsonOutput: input pool is invalid, contains key with no associated value");
    }
  }
  else {
    // we can make the assumption that this node has no value because the pool
    // doesn't not allow parent nodes to have values
    if (n->value != NULL) {
      throw EssentiaException(
          "JsonOutput: input pool is invalid, a parent key should not have a"
          "value in addition to child keys");
    }

    *s << " {\n";

    // and then emit the json for all of its children, recursive call
    for (int i=0; i<(int)n->children.size(); ++i) {
      emitJson(s, n->children[i], indent+"    ");
      if (i < (int)n->children.size()-1) {
          *s << ",";
      }
      *s << "\n";
    }

    *s << indent << "}";
  }
}


void outputYamlToStream(YamlNode& root, ostream* out) {
  for (int i=0; i<(int)root.children.size(); ++i) {
    *out << "\n";
    emitYaml(out, root.children[i], "");
  }
}


void outputJsonToStream(YamlNode& root, ostream* out) {
  *out << "{\n";
  for (int i=0; i<(int)root.children.size(); ++i) {
    emitJson(out, root.children[i], "");
    if (i < (int)root.children.size()-1) {
        *out << ",";
    }
    *out << "\n";
  }
  *out << "\n}";
}


void YamlOutput::outputToStream(ostream* out) {
  // set precision to be high enough
  out->precision(12);

  const Pool& p = _pool.get();

  // create the YamlNode Tree
  YamlNode root("doesn't matter what I put here, it's not getting emitted");

  // add metadata.version.essentia to the tree
  YamlNode* essentiaNode = new YamlNode("essentia");

  essentiaNode->value = new Parameter(essentia::version);

  YamlNode* versionNode = new YamlNode("version");
  versionNode->children.push_back(essentiaNode);

  YamlNode* metadataNode = new YamlNode("metadata");
  metadataNode->children.push_back(versionNode);

  root.children.push_back(metadataNode);

  // fill the YAML tree with the values form the pool
  fillYamlTree(p, &root);

  if (_outputJSON) {
      outputJsonToStream(root, out);
  } else {
      outputYamlToStream(root, out);
  }
}


void YamlOutput::compute() {
  if (_filename == "-") {
    outputToStream(&cout);
  }
  else {
    ofstream out(_filename.c_str());
    outputToStream(&out);
    out.close();

    if (_doubleCheck) {
      ostringstream expected;
      outputToStream(&expected);

      // read the file we just wrote...
      ifstream f(_filename.c_str());
      if (!f.good()) {
        throw EssentiaException("YamlOutput: error when double-checking the output file; it doesn't look like it was written at all");
      }
      ostringstream written;
      // we need to compare using streambuffers or otherwise
      // the check fails on windows due to new lines
      written << f.rdbuf();
      if (written.str() != expected.str()) {
        throw EssentiaException("YamlOutput: error when double-checking the output file; it doesn't match the expected output");
      }
    }
  }
}
