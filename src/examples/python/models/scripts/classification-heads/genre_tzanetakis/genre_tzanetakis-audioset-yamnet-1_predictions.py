from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictVGGish(graphFilename="audioset-yamnet-1.pb", input="melspectrogram", output="embeddings")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="genre_tzanetakis-audioset-yamnet-1.pb", output="model/Softmax")
predictions = model(embeddings)
