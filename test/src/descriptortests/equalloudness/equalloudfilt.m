function [a1,b1,a2,b2]=equalloudfilt(fs)
% Design a filter to match equal loudness curves
% 9/7/2001

% If the user hasn't specified a sampling frequency, use the CD default
if nargin<1,
   fs=44100;
end

% Specify the 80 dB Equal Loudness curve
if fs==44100 | fs==48000,
   EL80=[0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;4000,70.5;5000,74;6000,79;7000,84;8000,86;9000,86;10000,85;12000,95;15000,110;20000,125;fs/2,140];
elseif fs==32000,
   EL80=[0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;4000,70.5;5000,74;6000,79;7000,84;8000,86;9000,86;10000,85;12000,95;15000,110;fs/2,115];
else
   error('Filter not defined for current sample rate');
end

% convert frequency and amplitude of the equal loudness curve into format suitable for yulewalk
f=EL80(:,1)./(fs/2);
m=10.^((70-EL80(:,2))/20);

% Use a MATLAB utility to design a best bit IIR filter
[b1,a1]=yulewalk(10,f,m);

% Add a 2nd order high pass filter at 150Hz to finish the job
[b2,a2]=butter(2,(150/(fs/2)),'high');

