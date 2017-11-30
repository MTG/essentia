function y = flatness(x);

f = geomean(x) / mean(x);

c = max(x) / mean(x);

y = [f, c]