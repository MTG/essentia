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

#include "yamlinput.h"
#include "yamlast.h"
#include <fstream>

using namespace std;
using namespace essentia;
using namespace standard;

const char* YamlInput::name = "YamlInput";
const char* YamlInput::description = DOC("This algorithm deserializes a file formatted in YAML to a Pool. This file can be serialized back into a YAML file using the YamlOutput algorithm. See the documentation for YamlOutput for more information on the specification of the YAML file.\n"
"\n"
"Note: If an empty sequence is encountered (i.e. \"[]\"), this algorithm will assume it was intended to be a sequence of Reals and will add it to the output pool accordingly. This only applies to sequences which contain empty sequences. Empty sequences (which are not subsequences) are not possible in a Pool and therefore will be ignored if encountered (i.e. foo: [] (ignored), but foo: [[]] (added as a vector of one empty vector of reals).");

// takes an AST that's created by src/utils/essentiayaml and dumps it into a pool
void updatePool (const YamlNode* n, Pool* p, const string& keyPrefix);

// takes a YamlMappingNode and converts it to a StereoSample
StereoSample parseStereoSample(const YamlMappingNode& node);

void YamlInput::configure() {
  if (parameter("filename").isConfigured()) {
    _filename = parameter("filename").toString();
  }
}


// Computation is broken into two phases: the first phase utilizes lib yaml to
// parse the actual yaml file into an abstract syntax tree (AST), the second
// phase converts the AST into a Pool.
void YamlInput::compute() {
  if (!parameter("filename").isConfigured()) {
    throw EssentiaException("YamlInput: 'filename' parameter has not been configured");
  }

  Pool& p = _pool.get();

  FILE* file = fopen(_filename.c_str(), "rb");

  // check that the file exists:
  if (!file) throw EssentiaException("YamlInput: could not open file ", _filename);

  // First phase, build AST
  YamlNode* root = NULL;
  try {
    root = parseYaml(file);
  }
  catch (const YamlException& e) {
    if (fclose(file) != 0) {
      cout << "WARNING: YamlInput: an error occured while closing the yaml file" << endl;
    }
    throw EssentiaException("YamlInput: error during parsing: ", e.what());
  }

  if (fclose(file) != 0) {
    cout << "WARNING: YamlInput: an error occured while closing the yaml file" << endl;
  }

  YamlMappingNode* rootMap = dynamic_cast<YamlMappingNode*>(root);

  if (!rootMap) {
    throw EssentiaException("YamlInput: root node is not a mapping node, yaml "
                            "was not generated from a valid Pool");
  }

  // second phase, convert AST to std::map
  updatePool(rootMap, &p, "");

  delete rootMap;
}

// this function converts the AST generated in the first phase of computation
// into a Pool
void updatePool(const YamlNode* n, Pool* p, const string& keyPrefix) {
  //cout << "Updating pool with prefix: " << keyPrefix << endl;
  // dispatch on node type (mapping, sequence, or scalar nodes)

  // mapping nodes
  if (const YamlMappingNode* mapNode = dynamic_cast<const YamlMappingNode*>(n)) {
    //cout << "node is a map" << endl;
    for (map<string, YamlNode*>::const_iterator it = mapNode->getData().begin();
         it != mapNode->getData().end();
         ++it) {
      updatePool(it->second, p, (keyPrefix=="")? it->first : keyPrefix + "." + it->first);
    }
    return;
  }

  // sequence nodes
  if (const YamlSequenceNode* seqNode = dynamic_cast<const YamlSequenceNode*>(n)) {
    //cout << "node is a sequence" << endl;
    const vector<YamlNode*>& data = seqNode->getData();

    // first, figure out what yaml node types are inside data

    // if empty, assume sequence of reals
    if (data.empty()) {
      vector<Real> values;
      p->set(keyPrefix, values);
      return;
    }

    // sequence of scalars
    if (const YamlScalarNode* firstSeqSclrElmt = dynamic_cast<const YamlScalarNode*>(data[0])) {
      // figure out what scalar types we have {string, Real}
      YamlScalarNode::YamlScalarType tp = firstSeqSclrElmt->getType();

      //   strings
      if (tp == YamlScalarNode::STRING) {
        for (int i=0; i<int(data.size()); ++i) {
          const YamlScalarNode* sclrElmt = dynamic_cast<const YamlScalarNode*>(data[i]);
          if (!sclrElmt || sclrElmt->getType() != tp) {
            throw EssentiaException("YamlInput: mixed sequence types are not supported");
          }
          p->add(keyPrefix, sclrElmt->toString());
        }
        return;
      }

      //   Reals
      if (tp == YamlScalarNode::FLOAT) {
        for (int i=0; i<int(data.size()); ++i) {
          const YamlScalarNode* sclrElmt = dynamic_cast<const YamlScalarNode*>(data[i]);
          if (!sclrElmt || sclrElmt->getType() != tp) {
            throw EssentiaException("YamlInput: mixed sequence types are not supported");
          }
          p->add(keyPrefix, sclrElmt->toFloat());
        }
        return;
      }

      throw EssentiaException("YamlInput: only string and Real data types are the only supported scalars within a sequence");
    }

    // sequence of sequences
    if (const YamlSequenceNode* nonEmptySubSeq = dynamic_cast<const YamlSequenceNode*>(data[0])) {
      // first subsequence empty
      if (nonEmptySubSeq->empty()) {
        // keep checking until we find a non-empty subsequence so we can
        // determine the innertype of the subsequences
        for (int i=1; i<int(data.size()); ++i) {
          const YamlSequenceNode* subSeq = dynamic_cast<const YamlSequenceNode*>(data[i]);
          if (!subSeq) {
            throw EssentiaException("YamlInput: mixed sequence types are not supported");
          }

          if (subSeq->empty()) continue;

          // found non-empty subseq
          nonEmptySubSeq = subSeq;
          break;
        }

        // if no non-empty subsequence found, add them all as vector<Real>s
        if (nonEmptySubSeq->empty()) {
          for (int i=0; i<int(data.size()); ++i) p->add(keyPrefix, vector<Real>());
          return;
        }
      }

      // at this point we have a nonEmptySubSeq which we can probe to figure out
      // the innertypes of the subseqs

      // scalar innertypes
      if (const YamlScalarNode* firstSubSeqSclr = dynamic_cast<const YamlScalarNode*>(nonEmptySubSeq->getData()[0])) {

        // string scalars
        if (firstSubSeqSclr->getType() == YamlScalarNode::STRING) {

          for (int i=0; i<int(data.size()); ++i) {
            const YamlSequenceNode* subSeq = dynamic_cast<const YamlSequenceNode*>(data[i]);
            if (!subSeq) {
              throw EssentiaException("YamlInput: mixed sequence types are not supported");
            }

            const vector<YamlNode*>& subData = subSeq->getData();
            vector<string> strVec(subData.size());
            for (int j=0; j<int(strVec.size()); ++j) {
              const YamlScalarNode* subSeqScalar = dynamic_cast<const YamlScalarNode*>(subData[j]);
              if (!subSeqScalar) {
                throw EssentiaException("YamlInput: mixed sub-sequence types are not supported");
              }

              strVec[j] = subSeqScalar->toString();
            }

            p->add(keyPrefix, strVec);
          }

          return;
        }

        if (firstSubSeqSclr->getType() == YamlScalarNode::FLOAT) {
          for (int i=0; i<int(data.size()); ++i) {
            const YamlSequenceNode* subSeq = dynamic_cast<const YamlSequenceNode*>(data[i]);
            if (!subSeq) {
              throw EssentiaException("YamlInput: mixed sequence types are not supported");
            }

            const vector<YamlNode*>& subData = subSeq->getData();
            vector<Real> realVec(subData.size());
            for (int j=0; j<int(realVec.size()); ++j) {
              const YamlScalarNode* subSeqScalar = dynamic_cast<const YamlScalarNode*>(subData[j]);
              if (!subSeqScalar) {
                throw EssentiaException("YamlInput: mixed sub-sequence types are not supported");
              }

              realVec[j] = subSeqScalar->toFloat();
            }

            p->add(keyPrefix, realVec);
          }
          return;
        }

        throw EssentiaException("YamlInput: only string and Real data types are supported as YamlScalarNodes");
      }

      // we have yet another sequence (a subsubsequence)
      if (dynamic_cast<const YamlSequenceNode*>(nonEmptySubSeq->getData()[0])) {
        for (int i=0; i<int(data.size()); ++i) {
          const YamlSequenceNode* subseq = dynamic_cast<const YamlSequenceNode*>(data[i]);

          if (!subseq) throw EssentiaException("YamlInput: mixed sub-sequence types are not supported");

          const vector<YamlNode*>& subdata = subseq->getData();
          const YamlSequenceNode* subsubseq = dynamic_cast<const YamlSequenceNode*>(subdata[0]);

          if (subsubseq->empty()) {
            throw EssentiaException("YamlInput: sequences of matrices that have at least one dimension equal to 0 are not permitted");
          }

          // get dimensions
          int d1 = subdata.size();
          int d2 = subsubseq->size();
          TNT::Array2D<Real> mat(d1, d2, 0.0);

          for (int j=0; j<d1; ++j) {
            subsubseq = dynamic_cast<const YamlSequenceNode*>(subdata[j]);

            if (!subsubseq) throw EssentiaException("YamlInput: mixed sub-sequence types are not supported");

            const vector<YamlNode*>& subsubdata = subsubseq->getData();

            if (int(subsubdata.size()) != d2) {
              throw EssentiaException("YamlInput: in sequences of matrices, each matrix must be rectangular");
            }

            for (int k=0; k<d2; ++k) {
              const YamlScalarNode* sclr = dynamic_cast<const YamlScalarNode*>(subsubdata[k]);

              if (!sclr || sclr->getType() != YamlScalarNode::FLOAT) {
                throw EssentiaException("YamlInput: sequences of matrices can only consist of Reals");
              }

              mat[j][k] = sclr->toFloat();
            }
          }

          p->add(keyPrefix, mat);
        }

        return;
      }

      throw EssentiaException("YamlInput: unsupported YamlNode type encountered, within a subsequence");
    }

    // looks like a vector of StereoSamples (since StereoSamples are
    // represented as mapping nodes)
    if (dynamic_cast<const YamlMappingNode*>(data[0])) {
      // all elements in the sequence should be YamlMappingNodes
      for (int i=0; i<int(data.size()); ++i) {
        const YamlMappingNode* elmt = dynamic_cast<const YamlMappingNode*>(data[i]);
        if (!elmt) {
          throw EssentiaException("YamlInput: mixed sequence types are not supported");
        }

        p->add(keyPrefix, parseStereoSample(*elmt));
      }

      return;
    }

    throw EssentiaException("YamlInput: unsupported YamlNode type encountered, within a sequence");
  }

  // scalar nodes
  if (const YamlScalarNode* sclrNode = dynamic_cast<const YamlScalarNode*>(n)) {
    if (keyPrefix == "") {
      throw EssentiaException("YamlInput: adding a root level scalar, this shouldn't happen in valid YAML");
    }

    switch (sclrNode->getType()) {
      case YamlScalarNode::FLOAT:  p->set(keyPrefix, sclrNode->toFloat()); break;
      case YamlScalarNode::STRING: p->set(keyPrefix, sclrNode->toString()); break;
      default:
        throw EssentiaException("YamlInput: unsupported YamlScalarNode type encountered, expecting Reals or strings");
    }

    return;
  }

  throw EssentiaException("YamlInput: unsupported YamlNode type encountered");
}

// helper function used in updatePool to parse a StereoSample from a
// YamlMappingNode. Does some validation and throws an exception if
// given a mapping node that is not a valid StereoSample
StereoSample parseStereoSample(const YamlMappingNode& node) {
  // validate mapping node
  if (node.size() != 2) {
    throw EssentiaException("YamlInput: invalid StereoSample format--mapping node should consist of only 2 pairs, contains ", node.size());
  }

  // check if contains 'left' and 'right' keys
  if (node.getData().find("left") == node.getData().end() ||
      node.getData().find("right") == node.getData().end()) {
    throw EssentiaException("YamlInput: invalid StereoSample format--mapping node should contain the keys 'left' and 'right'");
  }

  // make sure values of 'left' and 'right' are scalar nodes
  YamlScalarNode* leftNode = dynamic_cast<YamlScalarNode*>(node.getData().find("left")->second);
  YamlScalarNode* rightNode = dynamic_cast<YamlScalarNode*>(node.getData().find("right")->second);

  if (leftNode == NULL || rightNode == NULL) {
    throw EssentiaException("YamlInput: invalid StereoSample format--the keys 'left' and 'right' must have scalare nodes as their values");
  }

  // make sure scalar values are of type Real
  if (leftNode->getType() != YamlScalarNode::FLOAT ||
      rightNode->getType() != YamlScalarNode::FLOAT) {
    throw EssentiaException("YamlInput: invalid StereoSample format--the keys 'left' and 'right' must have scalare nodes as their values which are Reals");
  }

  StereoSample result;
  result.left() = leftNode->toFloat();
  result.right() = rightNode->toFloat();

  return result;
}
