from essentia.standard import MonoLoader, TensorflowPredictVGGish

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictVGGish(graphFilename="audioset-yamnet-1.pb", input="melspectrogram", output="embeddings")
embeddings = model(audio)
