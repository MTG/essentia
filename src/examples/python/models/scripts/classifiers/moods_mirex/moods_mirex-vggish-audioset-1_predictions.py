from essentia.standard import MonoLoader, TensorflowPredictVGGish

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictVGGish(graphFilename="moods_mirex-vggish-audioset-1.pb", output="model/Softmax")
predictions = model(audio)
