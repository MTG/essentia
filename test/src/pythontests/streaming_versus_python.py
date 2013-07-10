# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

#! /usr/bin/env python
import os
import sys
import yaml
import numpy


###########################################################################################
# define variables. You should only change TOP and AUDIO_FILENAME
TOP = "../../../"
#AUDIO_FILENAME = TOP + "../audio/recorded/britney.wav"
PYTHON_DIR = TOP + "src/python"
PROFILE_DIR = TOP + "src/python/essentia/profiles"
PROFILE = "all_config.yaml"
PY_EXTRACTOR = 'essentia_extractor' #PYTHON_DIR + '/essentia_extractor'
C_EXTRACTOR = TOP + "src/examples/streaming_extractor"
# define variable for errors:
EPSILON = 1e-6
###########################################################################################

#sys.path.append( PYTHON_DIR )
#from essentia import EssentiaError

def EqualNumbers( a, b, epsilon ):
    if a == b: return 1
    if a and b and a/b < 0:
      return -1 # different sign
    a = abs(a)
    b = abs(b)
    error = abs( a -b )/a
    if error <= epsilon:
     # print_green("True\t" + str(error))
      return 1 # considered equal
    else:
     # print_red("False\t" + str(error))
      return 0 # considered different

def EqualList( a, b ):
    if not a and not b:
       return 1
    if len(a) != len(b): return 0 # this has already been checked anyway. redundant!!
    for i in range(len(a)):
        if( type(a[i]) != type(b[i]) ): return -2
        else:
          if type(a[i]) == float:
              if EqualNumbers( a[i], b[i], EPSILON ) !=1: return 0
          elif type(a[i]) == list:
              if EqualList(a[i], b[i] ) != 1: return 0
          elif type(a[i]) == dict:
              if EqualDict(a[i], b[i] ) != 1: return 0
          elif isinstance(a[i], (str, int)):
              if a[i] != b[i]: return 0
    return 1

def Equal( a, b ):
    if type(a) != type(b):
        if isinstance(a, (int,float)) and  isinstance(b, (int,float)):
            pass # in case of comparing 1 with 1.0 we want to get ok, and not failed
        else:
            print "ERROR: comparing", a, b, "which are of different type"
            return -2
    if isinstance(a, (str,int)):
       if a == b : return 1
    elif type(a) == float:
      return EqualNumbers( a, b, EPSILON )
    elif type(a) == list:
      return EqualList( a, b )
    else:
      print "ERROR: Type not supported for comparison"
      return -3

def EqualDict( d1, d2 ):
    failed ={}
    for k, v in d1.items():
      if Equal(d1[k], d2[k]) != 1:
        failed[k]=(d1[k], d2[k])
    return failed


def compare_dict_different_lengths( d1, d2 ):
    failed ={}
    for stat in d1.keys():
      if stat in d2:
          if Equal( d1[stat], d2[stat] ) != 1:
              failed[stat] = (d1[stat], d2[stat])
      else:
          failed[stat] = (d1[stat], "None")
    for stat in d2.keys():
        if stat not in d1:
          failed[stat] = ("None", d2[stat])
    return failed


###########################################################################################

def run_python_extractor( py_extractor ):
    if(py_extractor == ''):
        sys.exit(1)
    try:
      os.system(py_extractor)
    except( essentia.EssentiaError, RuntimeError ):
      print 'ERROR:', sys.exc_type, sys.exc_value
    return

def run_cpp_extractor( cpp_extractor ):
    if( cpp_extractor == ''):
        print "Error running cpp_extractor\n"
        sys.exit(1)
    try:
      os.system(cpp_extractor)
    except( essentia.EssentiaError, RuntimeError ):
      print 'ERROR:', sys.exc_type, sys.exc_value
    return

###########################################################################################

def compare_namespaces( c_data ={}, py_data={}, c_file='', py_file='' ):
    c_ns = c_data.keys() #namespaces computed by cpp
    py_ns = py_data.keys() #namespaces computed by python
    c_ns.sort()
    py_ns.sort()

    not_in_c=[] # namespaces found in python file but not in c++ file
    not_in_py=[] # namespaces found in c++ file but not in python file
    if c_ns != py_ns :
      for ns in c_ns:
        if ns not in py_ns:
          not_in_py.append(ns)
      for ns in py_ns:
        if ns not in c_ns:
          not_in_c.append(ns)
    return not_in_c, not_in_py

###########################################################################################

def delete_from_list( l, val ):
    if not l:
      print "Error, trying to delete from an empty list"
      return
    while val in l:l.remove(val)
    return

def dict2list( adict ):
    for key in sorted(adict.keys()):
        if isinstance( adict[key], dict ):
            for item in dict2list(adict[key]):
                yield [key] + item
        else: yield [[key] + [adict[key]]]
###########################################################################################

def compare_descriptors( c_data ={}, py_data={} ):
    result={}
    not_in_py=[] # descriptors found in c++ file but not in python file
    not_in_c=[] # descriptors found in python file but not in c++ file
    for ns in c_data.keys():
      if ns != "metadata" and ns in py_data.keys():
        for desc, c_values in c_data[ns].items():
            if py_data[ns].has_key(desc):
              c_len = len(c_data[ns][desc].values())
              if py_data[ns][desc] is None:
                py_len = 0
              else:
                py_len = len(py_data[ns][desc].values())
              if c_len != py_len:
                if py_len and c_len:
                  #result[desc]=(c_data[ns][desc].values(), py_data[ns][desc].values())
                  result[desc]=compare_dict_different_lengths( c_data[ns][desc], py_data[ns][desc])
                elif py_len:
                  result[desc]=( 'None', py_data[ns][desc].values())
                else:result[desc]=( c_data[ns][desc].values(), 'None')
              else:
                py_values = py_data[ns][desc]
                result[desc] = EqualDict(c_values, py_values)
            else:
             # print "ERROR: " + desc + " not found in python file"
              not_in_py.append(desc)
        for desc in py_data[ns].keys():
           if desc not in c_data[ns].keys():
             # print "ERROR: " + ns + "::" + desc + " not found in C++ file"
              not_in_c.append( desc )

    # METADATA STUFF. This should be implemented more cleanly and smarter. As it is
    # now is ugly and very ad-hoc:
      elif ns == "metadata":
        pass # let's skip this for now. However the code below works, we need to have
             # same output format for python and c++, otherwise parsing is ugly
      #    #positive_meta, negative_meta, meta_not_c, meta_not_py = compare_metadata(c_data[ns], py_data[ns])
      #  audio_properties=c_data[ns]["audio_properties"]
      #  # get tags from c++ metadata:
      #  tags=c_data[ns]['tags']
      #  for k in py_data[ns].keys():
      #    if k != 'version':
      #      py_value = py_data[ns][k]["value"]
      #      if k in audio_properties:
      #        if Equal(audio_properties[k], py_value) != 1:
      #          result[k] = (audio_properties[k], py_value )
      #      elif k in tags:
      #          if Equal(tags[k], py_value) != 1:
      #            result[k] = (tags[k], py_value )
      #      else: # not found in c++
      #          not_in_c.append( k )
      #    else:
      #        if Equal(c_data[ns]['version']['essentia'],\
      #                py_data[ns]['version']['essentia']) != 1:
      #          result[k]=(c_data[ns]['version']['essentia'], py_data[ns]['version']['essentia'])
      #  for k in audio_properties.keys():
      #     if k not in py_data[ns]: not_in_py.append(k)
      #  for k in tags.keys():
      #     if k not in py_data[ns]: not_in_py.append(k)

    return result, not_in_c, not_in_py

###########################################################################################

def compare( c_file='', py_file='', log_file='' ) :
    try:
      c_data = yaml.safe_load(open(c_file).read())
    except yaml.YAMLError, exception:
      print "Error in " + c_file + " : " + exception
    try:
      py_data = yaml.safe_load(open(py_file).read())
    except yaml.YAMLError, exception:
      print "Error in " + py_file + " : " + exception
    if c_data =={} or py_data == {}:
      print "\tERROR: passing empty files for comparison"
      sys.exit(1)
    desc_result, desc_not_in_c, desc_not_in_py = compare_descriptors( c_data, py_data )
    positive_test = []
    negative_test = []
    for k,v in desc_result.items():
        if v == {}:positive_test.append( k )
        else: negative_test.append( k )
    n_positive = len(positive_test)
    n_negative = len(negative_test)
    print "\n*******************************************************************"
    print "TOTAL TESTS: " , n_positive+n_negative
    print_green( "\tPOSITIVE_TESTS: " + str(n_positive) + "\n\t\t" + str(positive_test) )
    print_red( "\tNEGATIVE_TESTS: " + str(n_negative) + "\n\t\t" + str(negative_test) )
    print_yellow( "\tUNABLE TO TEST: " + str(len(desc_not_in_c) + len(desc_not_in_py)))
    FILE = open( log_file, 'w' )
    if n_negative:
      FILE.write("\n#*******************************************************************")
      FILE.write("\n#TOTAL TESTS: " + str(n_positive + n_negative) )
      FILE.write("\n#\tPOSITIVE_TESTS: " + str(n_positive) + "\n#\t\t" + str(positive_test) )
      FILE.write("\n#\tNEGATIVE_TESTS: " + str(n_negative) + "\n#\t\t" + str(negative_test) )
      FILE.write("\n#\tUNABLE TO TEST: " + str( len(desc_not_in_c) + len(desc_not_in_py) ) )
    if len( desc_not_in_c):
        print_yellow( "\n\treason: not found in the C++ file:")
        print_yellow( desc_not_in_c )
        FILE.write("\n#\treason: not found in the C++ file:")
        FILE.write( str(desc_not_in_c ) )
    if len( desc_not_in_py):
        print_yellow( "\n\treason: not found in the Python file:")
        print_yellow( desc_not_in_py )
        FILE.write("\n#\treason: not found in the Python file:")
        FILE.write( str(desc_not_in_py) )
    FILE.write("\n#*******************************************************************")
    FILE.write("\n#The following tests did not met the requirements for equality with epsilon " + str(EPSILON))
    FILE.write("\n#Errors are presented as descriptor_name: c_value, python_value\n")
    for k, v in desc_result.items():
      if not v:
        FILE.write("\n" + str(k) + ":\n")
        if not isinstance( v, dict ) :
          FILE.write("\t\t" + str(v) + "\n" )
        else:
            for k1, v1 in v.items():
              val = str(v1).split('(')[1]
              val = val.split(')')[0]
              FILE.write("    " + str(k1) + ":    " + val + "\n" )

    # check if some namespaces where missing from any of the files:
    ns_not_in_c, ns_not_in_py = compare_namespaces( c_data, py_data, c_file, py_file )
    if len(ns_not_in_c) or len(ns_not_in_py):
        if not FILE : FILE = open( log_file, 'w' )
        print "\n#*******************************************************************"
        FILE.write("\n#*******************************************************************")
        print '''WARNING: Some namespaces could not be compared as they could
not be found in either ''' + c_file + " or " + py_file
        FILE.write('''\n# WARNING: Some namespaces could not be compared as they could
# not be found in either ''' + c_file + " or " + py_file )
        if ns_not_in_py:
            print "\tnot found in " + py_file + " : "
            FILE.write( "\n#\tnot found in " + py_file + " : " )
            for i in range(len(ns_not_in_py)):
                print_red( "\t* " + str(ns_not_in_py[i]) )
                File.write( "\n#\t* " + str(ns_not_in_py[i]) )
        if ns_not_in_c:
            print "\tnot found in " + c_file + " : "
            FILE.write("\n#\tnot found in " + c_file + " : " )
            for i in range(len(ns_not_in_c)):
                print_red( "\t* " + str(ns_not_in_c[i]))
                FILE.write( "\n#\t* " + str(ns_not_in_c[i]))
        print "*******************************************************************\n"
        FILE.write("\n#*******************************************************************")
    print "\nsee file", log_file, "for more detais\n"

###########################################################################################
def print_green( s = ''):
    print "\033[32;1m", s, "\033[0m"
    return

def print_red( s = ''):
    print "\033[31;1m", s, "\033[0m"
    return

def print_yellow( s =''):
    print "\033[93;1m", s, "\033[0m"
    return
###########################################################################################

def get_log_filename( s ):
    if s == '':
      print "Error: passing an empty filename"
      sys.exit(1)
    filename = s.split('.')[-2]
    if "/" in filename:
      filename = filename.split('/')[-1]
    filename +="_log.yaml"
    return filename

###########################################################################################


def run( audio_filename ):
    # Python extractor
    print '\nPython Extractor'
    py_file = 'python.yaml'
    run_python_extractor(PY_EXTRACTOR + " " + PROFILE_DIR + "/" + PROFILE + " " + audio_filename + " " + py_file )


    # C++ extractor
    print '\nC++ Extractor'
    c_file = 'c.yaml'
    run_cpp_extractor( C_EXTRACTOR + " " + audio_filename + " " + c_file )

    print
    # Comparison
    compare( c_file, py_file, get_log_filename( audio_filename ) )

if __name__ == '__main__':
  from optparse import OptionParser
  parser = OptionParser()
  opt, args = parser.parse_args()
  if( len(args) < 1 ) :
    print "usage: ./streaming_versus_python.py audio_filename"
    sys.exit(1)
  run( args[0] )
