function y = highfrequencycontent(x,sr)

% Masri method
hfc1 = 0;

bin2hz = (sr/2)/(length(x)-1);

for i=1:length(x)
    hfc1 = hfc1 + (i-1)*bin2hz  * x(i)*x(i);
end;

% Jensen method
hfc2 = 0;

for i=1:length(x)
    hfc2 = hfc2 + (i-1)*bin2hz * (i-1)*bin2hz * abs(x(i));
end;

% Brossier method
hfc3 = 0;

for i=1:length(x)
    hfc3 = hfc3 + (i-1)*bin2hz * abs(x(i));
end;

y = [hfc1 hfc2 hfc3];
