from essentia.standard import MonoLoader, TensorflowPredictMAEST

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictMAEST(graphFilename="discogs-maest-10s-fs-2.pb", output="PartitionedCall/Identity_7")
embeddings = model(audio)
