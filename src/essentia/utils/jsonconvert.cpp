/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "jsonconvert.h"

using namespace std;
using namespace essentia;

void JsonConvert::skipSpaces() {
  while (_pos < _size) {
    if (_str[_pos] == ' ' || _str[_pos] == '\n' || 
                             _str[_pos] == '\r' || _str[_pos] == '\t') {
      _pos++;
    }
    else 
      break;
  }
}

int JsonConvert::countBackSlashes() {
  // counts a number of '\' chars following together right before the current position
  int i = _pos - 1;

  while(i >= 0 && _str[i] == '\\') {
    i--;
  }
  return -_pos - 1 - i;
}



string JsonConvert::parseStringValue() {

  if (_pos == _size || _str[_pos] != '"') {
    throw JsonException("Error parsing json string value");
  }
  _pos++;
  string value;

  while(_pos < _size) {
    if (_str[_pos] == '"' && !(countBackSlashes() % 2)) {
      // even number of '\' corresponds to a quote char that is closing the string
      // uneven number corresponds to the quote char being part of the escaped string
      _pos++;
      return value;
    }
    else {
      value += _str[_pos];
      _pos++;
    }
  }

  throw JsonException("Error parsing json string value: achieved EOF");
}


string JsonConvert::parseNumValue() {
  string value;

  while(_pos < _size) {
    if (_str[_pos] == ',' || _str[_pos] == '}' || _str[_pos] == ' ' || 
                             _str[_pos] == '\n' || _str[_pos] == '\r' || 
                             _str[_pos] == '\t' || _str[_pos] == ']') {
      break;
    }
    else {
      value += _str[_pos];
      _pos++;
    }
  }
  return value;
}


string JsonConvert::parseListValue() {

  if (_pos == _size || _str[_pos] != '[') {
    throw JsonException("Error parsing json list");
  }
  _pos++;
  string value("[");

  while(true) {
    skipSpaces();
    if (_pos == _size) {
      throw JsonException("Error parsing json list; achieved EOF");
    }
  
    if (_str[_pos] == ']') {  // end of list
      _pos++;
      value += "]";
      break;
    }
    else if (_str[_pos] == '"') { // string
      value += '"' + parseStringValue() + '"';
    }
    else if (_str[_pos] == '[') { // nested list
      value += parseListValue();
    }
    else if (_str[_pos] == '{') { 
      // TODO add support for dict elements in the list, yaml output should be 
      // like: [{left: 3, right: 6}, {left: 4, right: 7}]
      throw JsonException("Error parsing json list: dict elements are not supported");
    }
    else { // otherwise assume it is a number, but we do no check for incorrect data
      value += parseNumValue();
    }
    
    // expecting "," or "]"
    skipSpaces();

    if (_pos == _size || (_str[_pos] != ',' && _str[_pos] != ']')) {
      throw JsonException("Error parsing json list");
    }

    if (_str[_pos] == ',') {
      value += ", ";
      _pos++;
    }
  }

  return value;
}


string JsonConvert::parseDictKeyAndValue(const int& level) {

  string result(4 * level, ' '); // '    ' * 4

  // search for opening '"'
  skipSpaces();
  if (_pos == _size) {
    throw JsonException("Error parsing json dictionary: unexpected EOF");
  }

  if (_str[_pos] == '}') {
    throw JsonException("Error parsing json dictionary: emtpy dictionaries are not supported");   
  }
  else if (_str[_pos] != '"') {
    throw JsonException("Error parsing json dictionary: expected a string value as the key");
  }

  // extract key value
  string key = parseStringValue();

  skipSpaces();
  if (_pos == _size || _str[_pos] != ':') {
    throw JsonException("Error parsing json dictionary: ':' was expected");
  }
  _pos++;
  skipSpaces();

  // extract value 
  if (_pos==_size) {
    throw JsonException("Error parsing json dictionary: unexpected EOF");
  }
  if (_str[_pos] == '}') {
    throw JsonException("Error parsing json dictionary: missing value associated with the key");
  }

  if (_str[_pos] == '{') {  // nested dict
    result += key + ":" + "\n";
    result += parseDict(level+1);
  }
  else if (_str[_pos] == '[') { // list
    result += key + ": " + parseListValue() + '\n';
  }
  else if (_str[_pos] == '"') { // string
    result += key + ": \"" + parseStringValue() + "\"\n";
  }
  else { // numerical value
    result += key + ": " + parseNumValue() + '\n';
  }

  skipSpaces();

  // expecting ',' or '}'
  if (_pos==_size) {
    throw JsonException("Error parsing json dictionary: unexpected EOF");  
  }
  if (_str[_pos] != '}' && _str[_pos] != ',') {
    throw JsonException("Error parsing json dictionary: expecting '}' or ','");  
  }

  return result;
}


string JsonConvert::parseDict(const int& level) {
  string result;

  skipSpaces();
  if (_pos==_size) {
    throw JsonException("Error parsing json dictionary: unexpected EOF");  
  }
  if (_str[_pos] != '{') {
    throw JsonException("Error parsing json dictionary: expected '}'");  
  }

  while(true) {
    _pos++; // ','
    result += parseDictKeyAndValue(level);
    if (_str[_pos] == '}') {
      _pos++;
      break;
    }
  }
  
  skipSpaces();
  if (level==0 && _pos != _size) {
    throw JsonException("Error parsing json dictionary: extra data after the root dictionary");  
  }

  return result;
}


string JsonConvert::convert() {
  return _result;
}
