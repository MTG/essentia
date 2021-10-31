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

import essentia.standard
import tempfile
import numpy as np

def test_audiofile(filename=None, type='sin440', duration=1):

    """Create a dummy wav audio file.
    This can be useful for tests that require loading audio files.
    
    Args:
        filename (string): Filename (default=None). If not specified, a 
            named temporary file will be created. 
        type (string): The type of audio to generate: 'silence' or 'sin440' 
            (default)
        duration (float): duration of audio in seconds (default 1 sec.)
    Returns:
        (string): Name of a temporary audio file
    """
    if type == 'sin440':
        samples = np.sin(2*np.pi*np.arange(44100*duration)*440/44100).astype(np.float32)
    elif type == 'silence':
        samples = [0] * int(44100 * duration)
    else:
        raise (ValueError, 'Wrong audio type:', type)
    
    if not filename:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf.close()
        filename = tf.name

    essentia.standard.MonoWriter(filename=filename)(samples)
    return filename
