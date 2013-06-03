function y = energybandrelation(x, startCutoffFrequency, stopCutoffFrequency, samplerate)

x_2 = power(x,2);

start = floor(startCutoffFrequency * (length(x) - 1) / ((samplerate / 2) + 0.5)) + 1;
stop = floor(stopCutoffFrequency * (length(x) - 1) / ((samplerate / 2) + 0.5)) + 1;

y = sum(x_2(start:stop)) / sum(x_2);
