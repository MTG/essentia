function y = basicdesc(x,p)

energy = sum(power(x,2));

rms = sqrt(energy/length(x));

powermean = power(sum(power(x,p))/length(x),1/p);

y = [mean(x) geomean(x) rms powermean median(x) var(x,1) energy];