# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

class WekaFile:
	""" an example of creating a weka file:
	weka_file = WekaFile("key_correct")
	weka_file.add_attribute("strength", "REAL")
	weka_file.add_attribute("first_to_second", "REAL")
	weka_file.add_attribute("scale_correct", ["true", "false"])
	weka_file.add_attribute("key_correct", ["true", "false"])
	weka_file.add_attribute("key", ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
	weka_file.add_attribute("scale", ["minor", "major"])
	weka_file.add_data([0.626103, 0.255083], ["true", "true", "F", "minor"])
	weka_file.write("weka_test.arff")
	"""

	def __init__(self, relation_name):
		self.relation_name = relation_name
		self.attributes = []
		self.data = []

	def add_attribute(self, attribute_name, values):
		""" append an attribute and it's value-type
		the values should either be a string (in case of a REAL for example)
		or a list of strings, for enum types """
		
		self.attributes.append((attribute_name, values))

	def add_data(self, data, truths):
		""" append a data-line to the weka-file
		data should be a list of values (probably floats), and truths
		should be the list of ground-truths. Notice that these should be strings. """
		
		self.data.append((data, truths))

	def write(self, filename):
		""" write the weka-file to disk """
		
		weka_file = open(filename, "w")
		
		# relation name
		weka_file.write("@RELATION " + self.relation_name + "\n")
		
		#atributes
		for (attribute_name, values) in self.attributes:
			weka_file.write("@ATTRIBUTE \'" + attribute_name + "\' ")
			if isinstance(values,list):
				weka_file.write("{" + ", ".join([str(value) for value in values]) + "}")
			else:
				weka_file.write(str(values))
			weka_file.write("\n")
			
		# data
		weka_file.write("@DATA\n")
		for (data, truths) in self.data:
			data_part = " ".join([str(value) for value in data])
			truth_part = ", ".join([str(value) for value in truths])
			weka_file.write(data_part + ", " + truth_part + "\n")

if __name__ == "__main__":
	"""a small unit-test to see if this thing actually works..."""
	
	weka_file = WekaFile("key_correct")
	weka_file.add_attribute("strength", "REAL")
	weka_file.add_attribute("first_to_second", "REAL")
	weka_file.add_attribute("scale_correct", ["true", "false"])
	weka_file.add_attribute("key_correct", ["true", "false"])
	weka_file.add_attribute("key", ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
	weka_file.add_attribute("scale", ["minor", "major"])
	weka_file.add_data([0.626103, 0.255083], ["true", "true", "F", "minor"])
	weka_file.write("weka_test.arff")

	expected_output = """@RELATION key_correct
@ATTRIBUTE strength REAL
@ATTRIBUTE first_to_second REAL
@ATTRIBUTE scale_correct {true, false}
@ATTRIBUTE key_correct {true, false}
@ATTRIBUTE key {A, A#, B, C, C#, D, D#, E, F, F#, G, G#}
@ATTRIBUTE scale {minor, major}
@DATA
0.626103 0.255083, true, true, F, minor
"""
	output = "".join(file("weka_test.arff").readlines())
	
	if output == expected_output:
		print "test succeeded"
	else:
		print "test failed"
		print "---------output-------------------------"
		print output
		print "---------expected output----------------"
		print expected_output
