from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictMusiCNN(graphFilename="genre_tzanetakis-musicnn-mtt-2.pb")
predictions = model(audio)
