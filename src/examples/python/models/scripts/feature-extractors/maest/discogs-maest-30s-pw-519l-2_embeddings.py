from essentia.standard import MonoLoader, TensorflowPredictMAEST

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictMAEST(graphFilename="discogs-maest-30s-pw-519l-2.pb", output="PartitionedCall/Identity_7")
embeddings = model(audio)
