#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Created by pablo on 31/01/18

"""
Class for event tests. Here the methods can be overrided to fit the 'values' case
"""


from qa_test import *


class QaTestValues(QaTest):
    def __init__(self, *args, **kwargs):
        # use QaTest constructor with hardcoded `test_type`
        if len(args) > 1:
            args = args[1:]
        kwargs.pop('test_type', None)

        QaTest.__init__(self, 'events', *args, **kwargs)

        self.add_error()

    def add_error(self):
        class Error(QaMetric):
            """
            Simple elementwise error
            """
            def score(self, reference, estimated):
                scores = np.abs(reference - estimated)
                return scores

        self.set_metrics(Error())
