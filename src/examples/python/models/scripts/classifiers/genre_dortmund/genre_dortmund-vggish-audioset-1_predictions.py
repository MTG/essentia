from essentia.standard import MonoLoader, TensorflowPredictVGGish

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictVGGish(graphFilename="genre_dortmund-vggish-audioset-1.pb")
predictions = model(audio)
