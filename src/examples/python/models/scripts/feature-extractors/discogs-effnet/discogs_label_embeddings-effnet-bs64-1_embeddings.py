from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictEffnetDiscogs(graphFilename="discogs_label_embeddings-effnet-bs64-1.pb", output="PartitionedCall:1")
embeddings = model(audio)
