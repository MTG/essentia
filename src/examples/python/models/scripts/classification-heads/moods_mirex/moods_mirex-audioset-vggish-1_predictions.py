from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictVGGish(graphFilename="audioset-vggish-3.pb", output="model/vggish/embeddings")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="moods_mirex-audioset-vggish-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
predictions = model(embeddings)
