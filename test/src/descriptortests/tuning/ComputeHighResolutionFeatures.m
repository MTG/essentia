function [equal_tempered_dev, nontempered2tempered_energy_ratio, nontempered2tempered_peaks_energy_ratio]=ComputeHighResolutionFeatures(hpcp) 
% Compute high-resolution chroma features
% Set of feature derived from computing a high-resolution HPCP vector
% (minimum size=120, 10 cents or more)

% 1.- Equal-temperament deviation: measure of the deviation of HPCP local maxima
%     with respect to equal-tempered bins. 
% Steps: 
% a) Compute local maxima of HPCP vector
% b) Compute the deviations from equal-tempered (abs) bins and average
%
% 2.- NonTempered2Tempered energy ratio: ratio betwen the energy on
%     non-tempered bins and the total energy, computed from the HPCP average
  
hpcpsize = length(hpcp); 
nPeaks = 24;
showplots = false; 

% 1.- Equal-temperament deviation: measure of the deviation of HPCP local maxima
%     with respect to equal-tempered bins. 
% a) Compute local maxima of HPCP vector
[ploc, pval]=DetectPeaks(hpcp, nPeaks);

%     plot peaks
aux=(1:(hpcpsize/12):hpcpsize);
if(showplots)
    figure(2)
    plot(hpcp);
    hold
    plot(ploc,pval,'*r');
    grid  
    set(gca,'xtick',aux);
    set(gca,'XTickLabel',{'A';'#';'B';'C';'#';'D';'#';'E';'F';'#';'G';'#';});
    title('HPCP local maxima');
    hold
end

% b) Compute the deviations from equal-tempered (abs) bins and average
bins_per_st = hpcpsize/12;
ploc_dev = (ploc-1)./(bins_per_st)+1;
ploc_dev = mod(ploc_dev,1);

%   find negative deviations
up_half = find(ploc_dev > 0.5);
ploc_dev(up_half) = ploc_dev(up_half) - 1;

%   weight deviations by its amplitud
equal_tempered_dev = sum(abs(ploc_dev).*pval)/sum(pval); % Second implementation: normalize by the sum of the amplitudes (pval)
equal_tempered_dev,

% 2.- NonTempered2Tempered energy ratio: ratio betwen the energy on
%     non-tempered bins and the total energy, computed from the HPCP average
tempered_energy = sum(hpcp(aux).* hpcp(aux));
total_energy = sum(hpcp.*hpcp);
if total_energy > 0
    nontempered2tempered_energy_ratio = (total_energy - tempered_energy)/total_energy;
else
    nontempered2tempered_energy_ratio = 0;
end

nontempered2tempered_energy_ratio,

% if nontempered2tempered_energy_ratio == 0
%     disp('0 energy ratio'); pause;
% end
% 
% if nontempered2tempered_energy_ratio == 1
%     disp('1 energy ratio'); pause;
% end
% 3.- NonTempered2Tempered peak energy ratio: ratio betwen the energy on
%     non tempered peaks and the total energy, computed from the HPCP average
tempered_peaks = find(ploc_dev==0);
tempered_peaks_energy = sum(pval(tempered_peaks).* pval(tempered_peaks));
total_peaks_energy = sum(pval.*pval);
if total_peaks_energy > 0
    nontempered2tempered_peaks_energy_ratio = (total_peaks_energy - tempered_peaks_energy)/total_peaks_energy;
else
    nontempered2tempered_peaks_energy_ratio = 0;
end

nontempered2tempered_peaks_energy_ratio,

% if nontempered2tempered_peaks_energy_ratio == 0
%     disp('0 peak energy ratio'); pause;
% end
% 
% if nontempered2tempered_peaks_energy_ratio == 1
%     tempered_peaks_energy
%     disp('1 peak energy ratio'); pause;
% end
