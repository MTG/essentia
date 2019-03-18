#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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


"""
Class for event tests. Here the methods can be overrided to fit the 'values' case.
"""


from qa_test import *


class QaTestValues(QaTest):
    def __init__(self, *args, **kwargs):
        # Use QaTest constructor with hardcoded `test_type`.
        if len(args) > 1:
            args = args[1:]
        kwargs.pop('test_type', None)

        QaTest.__init__(self, 'events', *args, **kwargs)

        self.add_error()

    def add_error(self):
        class Error(QaMetric):
            """
            Simple elementwise error.
            """
            def score(self, reference, estimated):
                scores = np.abs(reference - estimated)
                return scores

        self.set_metrics(Error())
