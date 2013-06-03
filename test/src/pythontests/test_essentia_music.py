#!/usr/bin/python


def test():
  try:
    import essentia
    import os
    #from essentia.extractor import essentia_music
    from essentia import essentia_extractor
    #options, args = essentia_extractor.parse_args()
    #exec('options = ' + str(options))
    essentia_extractor.compute('music', "../../audio/recorded/britney.wav","foo.sig")
    os.unlink("foo.sig")
    return 0 
  except:
    raise
    print "Failed to run essentia_music"
    return 1
