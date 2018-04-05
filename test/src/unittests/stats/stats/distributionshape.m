function Y = distributionshape(X, N)
% X = x-axis (time index, spectral bin/frequency)
% N = y-axis (signal amplitude, spectral magnitude)
% For instance, if N is a 1000 bins array you have to put X = [0 1 2 ... 999]

rawmom1 = sum(X .* N) / sum(N);
rawmom2 = sum((X.^2) .* N) / sum(N);
rawmom3 = sum((X.^3) .* N) / sum(N);
rawmom4 = sum((X.^4) .* N) / sum(N);
rawmom5 = sum((X.^5) .* N) / sum(N);

centroid = rawmom1;

centmom2 = sum(((X - centroid).^2) .* N) / sum(N);
centmom3 = sum(((X - centroid).^3) .* N) / sum(N);
centmom4 = sum(((X - centroid).^4) .* N) / sum(N);

spread   = centmom2;
skewness = centmom3 / spread^(3/2) ;
kurtosis = (centmom4 / spread^2) - 3;

Y = [rawmom1 rawmom2 rawmom3 rawmom4 rawmom5 centmom2 centmom3 centmom4 centroid spread skewness kurtosis];