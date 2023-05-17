from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="msd-msd-musicnn-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
predictions = model(embeddings)
