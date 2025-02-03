from essentia import Pool
from essentia.standard import MonoLoader, TensorflowPredictMAEST, TensorflowPredict

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
embedding_model = TensorflowPredictMAEST(graphFilename="discogs-maest-20s-pw-2.pb", output="PartitionedCall/Identity_12")
embeddings = embedding_model(audio)

pool = Pool()
pool.set("embeddings", embeddings)

model = TensorflowPredict(graphFilename="genre_discogs400-discogs-maest-20s-pw-1.pb", inputs=["embeddings"], outputs=["PartitionedCall/Identity_1"])
predictions = model(pool)["PartitionedCall/Identity_1"]
