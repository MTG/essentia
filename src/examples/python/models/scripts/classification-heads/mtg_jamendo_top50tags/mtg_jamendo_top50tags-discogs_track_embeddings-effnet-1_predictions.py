from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs_track_embeddings-effnet-bs64-1.pb", output="PartitionedCall:1")
embeddings = embedding_model(audio)

model = TensorflowPredict2D(graphFilename="mtg_jamendo_top50tags-discogs_track_embeddings-effnet-1.pb")
predictions = model(embeddings)
