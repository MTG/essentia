from essentia.standard import MonoLoader, PitchCREPE

audio = MonoLoader(filename="audio.wav", sampleRate=16000, resampleQuality=4)()
model = PitchCREPE(graphFilename="crepe-tiny-1.pb")
time, frequency, confidence, activations = model(audio)
