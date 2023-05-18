from essentia.standard import MonoLoader, TensorflowPredictFSDSINet

audio = MonoLoader(filename="audio.wav", sampleRate=22050, resampleQuality=4)()
model = TensorflowPredictFSDSINet(graphFilename="fsd-sinet-vgg41-tlpf-1.pb")
predictions = model(audio)
