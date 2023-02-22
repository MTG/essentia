from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictMusiCNN(graphFilename="mtt-vgg-1.pb")
embeddings = model(audio)
