from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="genre_dortmund-effnet-discogs-1.pb", output="model/Softmax")
predictions = model(embeddings)
