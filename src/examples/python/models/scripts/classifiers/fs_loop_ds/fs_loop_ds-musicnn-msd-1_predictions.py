from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

audio = MonoLoader(filename="audio.wav", sampleRate=16000)()
model = TensorflowPredictMusiCNN(graphFilename="fs_loop_ds-musicnn-msd-1.pb", output="model/Softmax")
predictions = model(audio)
