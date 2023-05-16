from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictVGGish(graphFilename="audioset-vggish-3.pb", output="model/vggish/embeddings")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="deam-audioset-vggish-1.pb", input="flatten_in_input", output="dense_out")
predictions = model(embeddings)
