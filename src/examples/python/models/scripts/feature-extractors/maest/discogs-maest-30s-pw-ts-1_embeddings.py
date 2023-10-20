from essentia.standard import MonoLoader, TensorflowPredictMAEST

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictMAEST(graphFilename="discogs-maest-30s-pw-ts-1.pb", output="StatefulPartitionedCall:7")
embeddings = model(audio)
