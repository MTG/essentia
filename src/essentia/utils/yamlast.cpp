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

#include "yamlast.h"
#include <stack>
#include <cstdlib>

using namespace std;
using namespace essentia;

YamlNode::~YamlNode() {}

YamlSequenceNode::~YamlSequenceNode() {
  for (int i=0; i<int(_data.size()); ++i) {
    delete _data[i];
  }
}

YamlMappingNode::~YamlMappingNode() {
  for (map<string, YamlNode*>::iterator it = _data.begin();
       it != _data.end();
       ++it) {
    delete it->second;
  }
}

// this function should be used before returning from parseYaml (by either a
// return statement or an exception). It makes sure to
// delete any residuals before returning. Don't use this function in the
// middle of valid execution, as it will delete things you need
void cleanState(stack<YamlNode*>* pendingNodes, stack<YamlScalarNode*>* tempKeys, yaml_parser_t* parser, yaml_event_t* event) {
  if (pendingNodes != NULL) {
    while (!pendingNodes->empty()) {
      delete pendingNodes->top();
      pendingNodes->pop();
    }
  }

  if (tempKeys != NULL) {
    while (!tempKeys->empty()) {
      delete tempKeys->top();
      tempKeys->pop();
    }
  }

  if (parser != NULL) yaml_parser_delete(parser);
  if (event != NULL) yaml_event_delete(event);
}


void throwParserError(yaml_parser_t*);


// parses the yaml file and returns the root of the parsed AST
YamlNode* essentia::parseYaml(FILE* file) {
  // make sure to delete this object with 'yaml_parser_delete', taken care of
  // by cleanState
  yaml_parser_t parser;
  yaml_event_t event;

  // Create the Parser object
  yaml_parser_initialize(&parser);

  // Set a file input
  yaml_parser_set_input_file(&parser, file);

  // prepare temporary data structures
  stack<YamlNode*> pendingNodes;
  int pendingMapCount = 0; // count of mapping nodes in 'pendingNodes'
  stack<YamlScalarNode*> tempKeys; // keys for mapping nodes

  // get the first parser event
  if (!yaml_parser_parse(&parser, &event)) throwParserError(&parser);

  // Read the event sequence
  while (event.type != YAML_STREAM_END_EVENT) {
    //cout << "EVENT(("<<event.type<<"))" << endl;
    switch (event.type) {

      case YAML_SCALAR_EVENT: {
        //cout << "received YAML_SCALAR_EVENT ";

        string rawData = string((const char*)event.data.scalar.value,
                                event.data.scalar.length);


        // check whether it's been explicitly forced into a string by quoting it
        bool isString = event.data.scalar.style == YAML_SINGLE_QUOTED_SCALAR_STYLE ||
                        event.data.scalar.style == YAML_DOUBLE_QUOTED_SCALAR_STYLE;

        // might be a boolean..
        if (!isString && ((rawData=="true") || (rawData=="false"))) {
          rawData = (rawData=="true") ? "1" : "0";
        };

        // otherwise, see if we can autoconvert it to a float, otherwise it's a string
        char* endptr;
        double dummy = strtod(rawData.c_str(), &endptr);
	(void)dummy;

        isString = isString || (endptr != (&rawData[0] + rawData.size()));


        if (pendingNodes.empty()) {
          //cout << "no parents" << endl;
          if (isString) {
            cleanState(&pendingNodes, &tempKeys, &parser, &event);
            return new YamlScalarNode(rawData);
          }
          else {
            cleanState(&pendingNodes, &tempKeys, &parser, &event);
            return new YamlScalarNode(float(atof(rawData.c_str())));
          }
        }

        // if pending node is a sequence
        if (YamlSequenceNode* seq = dynamic_cast<YamlSequenceNode*>(pendingNodes.top())) {
          //cout << "parent is sequence" << endl;
          if (isString) seq->add(new YamlScalarNode(rawData));
          else seq->add(new YamlScalarNode(float(atof(rawData.c_str()))));
          break;
        }

        // else if pending node is a mapping
        if (YamlMappingNode* map = dynamic_cast<YamlMappingNode*>(pendingNodes.top())) {
          //cout << "parent is mapping" << endl;
          if (int(tempKeys.size()) < pendingMapCount) {
            //cout << "found key" << endl;
            // this is a key, add to tempKeys
            tempKeys.push(new YamlScalarNode(rawData));
            break;
          }

          //cout << "finalized mapping pair" << endl;

          // not a key, this is a value, check if Real or string
          if (isString) {
            map->add(tempKeys.top()->toString(),
                     new YamlScalarNode(rawData));
          }
          else {
            map->add(tempKeys.top()->toString(),
                     new YamlScalarNode(float(atof(rawData.c_str()))));
          }

          // key will now be owned by map, delete tempKey
          delete tempKeys.top();
          tempKeys.pop();
          break;
        }

        // else throw an error
        cleanState(&pendingNodes, &tempKeys, &parser, &event);
        throw YamlException("parsed YamlScalarNode has parent "
                                "which is not a sequence or mapping");
      }

      case YAML_SEQUENCE_START_EVENT:
        //cout << "received YAML_SEQUENCE_START_EVENT" << endl;
        pendingNodes.push(new YamlSequenceNode());
        break;

      case YAML_SEQUENCE_END_EVENT: {
        // remove top node from pendingNodes and make sure its a sequence
        YamlNode* n = pendingNodes.top();
        pendingNodes.pop();
        YamlSequenceNode* seq = dynamic_cast<YamlSequenceNode*>(n);

        if (seq == NULL) {
          cleanState(&pendingNodes, &tempKeys, &parser, &event);
          throw YamlException("received YAML_SEQUENCE_END_EVENT but top of pendingNode stack is not a YamlSequenceNode");
        }

        if (pendingNodes.empty()) {
          cleanState(&pendingNodes, &tempKeys, &parser, &event);
          return seq;
        }

        // add sequence to parent
        // if pending node is a sequence
        if (YamlSequenceNode* parentSeq = dynamic_cast<YamlSequenceNode*>(pendingNodes.top())) {
          parentSeq->add(seq);
          break;
        }

        // else if pending node is a mapping
        if (YamlMappingNode* parentMap = dynamic_cast<YamlMappingNode*>(pendingNodes.top())) {
          if (int(tempKeys.size()) < pendingMapCount) {
            cleanState(&pendingNodes, &tempKeys, &parser, &event);
            throw YamlException("received YAML_SEQUENCE_END_EVENT when parent is a YamlMappingNode and has no key already set. Can't set key to a sequence, that's just silly.");
          }

          //cout << "    adding to map under key: " << tempKeys.top()->getData().toString() << endl;
          parentMap->add(tempKeys.top()->toString(), seq);
          delete tempKeys.top();
          tempKeys.pop();
          break;
        }

        // else throw an error
        cleanState(&pendingNodes, &tempKeys, &parser, &event);
        throw YamlException("parsed YamlSequenceNode has a parent which is not a sequence or mapping");
      }

      case YAML_MAPPING_START_EVENT:
        //cout << "received YAML_MAPPING_START_EVENT" << endl;
        pendingNodes.push(new YamlMappingNode());
        ++pendingMapCount;
        break;

      case YAML_MAPPING_END_EVENT: {
        //cout << "received YAML_MAPPING_END_EVENT" << endl;
        YamlNode* n = pendingNodes.top();
        pendingNodes.pop();
        --pendingMapCount;
        YamlMappingNode* map = dynamic_cast<YamlMappingNode*>(n);

        if (map == NULL) {
          cleanState(&pendingNodes, &tempKeys, &parser, &event);
          throw YamlException("received YAML_MAPPING_END_EVENT "
                                  "but top of pendingNode stack is not a YamlMappingNode");
        }

        if (pendingNodes.empty()) {
          yaml_parser_delete(&parser);
          return map;
        }

        // add mapping to parent
        // if pending node is a sequence
        if (YamlSequenceNode* parentSeq = dynamic_cast<YamlSequenceNode*>(pendingNodes.top())) {
          parentSeq->add(map);
          break;
        }

        // else if pending node is a mapping
        if (YamlMappingNode* parentMap = dynamic_cast<YamlMappingNode*>(pendingNodes.top())) {
          if (int(tempKeys.size()) < pendingMapCount) {
            cleanState(&pendingNodes, &tempKeys, &parser, &event);
            throw YamlException("received YAML_MAPPING_END_EVENT"
                                    " when parent is a YamlMappingNode and has no key already set. "
                                    "Can't set key to a mapping, that's just silly.");
          }

          parentMap->add(tempKeys.top()->toString(), map);
          delete tempKeys.top();
          tempKeys.pop();
          break;
        }

        // else throw an error
        cleanState(&pendingNodes, &tempKeys, &parser, &event);
        throw YamlException("parsed YamlMappingNode has a "
                                "parent which is not a sequence or mapping");
      }

      case YAML_NO_EVENT:
        //cout << "received YAML_NO_EVENT" << endl;
        //cout << "NO EVENT FINISH!!" << endl;
        break;

      case YAML_STREAM_START_EVENT:
      case YAML_DOCUMENT_START_EVENT:
        // don't do anything
        break;

      default:
        //cout << "received no recognized event: " << event.type << endl;
        cleanState(&pendingNodes, &tempKeys, NULL, &event);
        throwParserError(&parser);
        break;
    }

    // The application is responsible for destroying the event object
    yaml_event_delete(&event);

    // Get the next event
    if (!yaml_parser_parse(&parser, &event)) {
      cleanState(&pendingNodes, &tempKeys, NULL, &event);
      throwParserError(&parser);
    }
  }

  // clean up
  cleanState(&pendingNodes, &tempKeys, &parser, &event);
  throw YamlException("reached unexpected parsing state");
}

void throwParserError(yaml_parser_t* parser) {
  stringstream msg;

  switch (parser->error) {
    case YAML_MEMORY_ERROR:
      msg << "Memory error: Not enough memory for parsing";
      break;

    case YAML_READER_ERROR:
      if (parser->problem_value != -1) {
        msg << "Reader error: "<< parser->problem << ": #"
            << parser->problem_value << " at " << parser->problem_offset;
      }
      else {
        msg << "Reader error: " << parser->problem << " at "
            << parser->problem_offset;
      }
      break;

    case YAML_SCANNER_ERROR:
      if (parser->context) {
        msg << "Scanner error: " << parser->context << " at line "
            << parser->context_mark.line+1 << ", column "
            << parser->context_mark.column+1 << "\n" << parser->problem
            << " at line " << parser->problem_mark.line+1 << ", column "
            << parser->problem_mark.column+1;
      }
      else {
        msg << "Scanner error: " << parser->problem << " at line "
            << parser->problem_mark.line+1 << ", column "
            << parser->problem_mark.column+1;
      }
      break;

    case YAML_PARSER_ERROR:
      if (parser->context) {
        msg << "Parser error: " << parser->context << " at line "
            << parser->context_mark.line+1 << ", column "
            << parser->context_mark.column+1 << "\n" << parser->problem
            << " at line " << parser->problem_mark.line+1 << ", column "
            << parser->problem_mark.column+1;
      }
      else {
        msg << "Parser error: " << parser->problem << " at line "
            << parser->problem_mark.line+1 << ", column "
            << parser->problem_mark.column+1;
      }
      break;

    default:
      /* Couldn't happen. */
      msg << "Internal error in yaml parsing";
      break;
  }

  yaml_parser_delete(parser);
  throw YamlException(msg.str());
}
