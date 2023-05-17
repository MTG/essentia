from essentia.standard import MonoLoader, TempoCNN

audio = MonoLoader(filename="audio.wav", sampleRate=11025, resampleQuality=4)()
model = TempoCNN(graphFilename="deeptemp-k4-3.pb")
global_tempo, local_tempo, local_tempo_probabilities = model(audio)
