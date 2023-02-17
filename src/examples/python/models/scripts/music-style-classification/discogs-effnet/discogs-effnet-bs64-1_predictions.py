from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb")
predictions = model(audio)
