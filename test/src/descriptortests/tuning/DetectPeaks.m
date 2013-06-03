% Function for peak detection 
% Adaptations made to work with HPCP by emilia, 28-03-2007
function [ploc, pval]=DetectPeaks(spectrum, nPeaks)

% function DetectPeaks(spectrum)
% Inputs:
%     spectrum: dB spectrum magnitude (abs(fft(signal))
%     nPeaks: maximum number of peaks to pick
% Outputs:
% ploc: bin number of peaks  (if ploc(i)==0, no peak detected)
% pval: magnitude values of the peaks in dB 

pval = []; %ones(nPeaks,1)*-100;
ploc = []; %zeros(nPeaks,1);

% Compute derivate
minSpectrum = min(spectrum);
difference = diff([minSpectrum; spectrum; minSpectrum]);

% Compute peak locatios from derivate
size = length(spectrum);
% peak locations
iloc = find(difference(1:size)>= 0 & difference(2:size+1) <= 0);
% peak values
ival = spectrum(iloc);
p = 1;

while(max(ival)>-100)
  [maxi, l] = max(ival); % find current maximum, maximum value
  
  pval = [pval maxi];
  % Put maximum to -100 dB, so that it's not the maximum anymore
  ival(l) = -100;
  aux = iloc(l);
  ploc = [ploc aux]; % save value and location
  
%  ind = find(abs(iloc(l)-iloc) > minspace);
  % find peaks which are far away
 % if (isempty(ind))
 %   break % no more local peaks to pick
 % end
        
 % ival = ival(ind); % shrink peak value and location array
 % iloc = iloc(ind);
 p=p+1;

end
