#!/usr/bin/python


def test():
  try:
    import essentia
    return 0
  except ImportError:
    print "Failed to import essentia module"
    return 1
