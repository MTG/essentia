from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
embeddings = model(audio)
