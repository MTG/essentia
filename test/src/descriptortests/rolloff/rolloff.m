function y = rolloff(x,cutoff,samplerate)

cutoffEnergy = cutoff * sum(power(x,2))

energy = 0;
stop = 0;

rolloff = 0;

for i=1:length(x)
    if (stop == 0)
        energy = energy + x(i)*x(i);
        if (energy > cutoffEnergy)
            rolloff = i-1;
            stop = 1;
        end;
    end;
end;

y = rolloff * (samplerate / 2) / (length(x) - 1);