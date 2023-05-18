from essentia.standard import MonoLoader, TensorflowPredictFSDSINet

audio = MonoLoader(filename="audio.wav", sampleRate=22050, resampleQuality=4)()
model = TensorflowPredictFSDSINet(graphFilename="fsd-sinet-vgg42-aps-1.pb")
predictions = model(audio)
