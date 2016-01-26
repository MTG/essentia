% Trying to design a filter to match equal loudness curves
% David Robinson http://www.David.Robinson.org 10/07/2001

% This script draws some of the graphs on the following web page:
% http://www.David.Robinson.org/replaygain/equal_loudness.html


% Set sampling rate
fs=44100;

% Specify Equal Loudness amplitude response
if fs==44100 | fs==48000,
   EL80=[0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;4000,70.5;5000,74;6000,79;7000,84;8000,86;9000,86;10000,85;12000,95;15000,110;20000,125;fs/2,140];
end
if fs==32000,
   EL80=[0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;4000,70.5;5000,74;6000,79;7000,84;8000,86;9000,86;10000,85;12000,95;15000,110;fs/2,115];
end
if fs==8000,
   EL80=[0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;fs/2,70.5];
end
linewidth=2;	% width of lines to plot in graphs

% Plot Target Response (70 minus equal loudness response)
semilogx(EL80(:,1),70-EL80(:,2),'-','LineWidth',linewidth)

hold on	% plot all graphs on same axis

% Generate an impulse to filter in order to yield the long-term impulse response of each filter
a(1:10000)=zeros;
a(5000)=1;

% Design an "A" Weighting filter
[B,A] = adsgn(fs);
b=filter(B,A,a);
frequency_response=fft(b);
amplitude_response=20*log10(abs(frequency_response));
frequency_axis=(0:length(b)-1)*fs/length(b);
min_f=2;
max_f=fix(length(b)/2)+1;
% Plot "A" Weighting curve
plot(frequency_axis(min_f:max_f),amplitude_response(min_f:max_f),'y','LineWidth',linewidth)	% plot interpolated line

% Convert target frequency and amplitude data into format suitable for yulewalk function
f=EL80(:,1)./(fs/2);
m=10.^((70-EL80(:,2))/20);

% Design a 10 coefficient filter using "yulewalk" function
[By,Ay]=yulewalk(10,f,m);
c=filter(By,Ay,a);

frequency_response=fft(c);
amplitude_response=20*log10(abs(frequency_response));
frequency_axis=(0:length(c)-1)*fs/length(c);
min_f=2;
max_f=fix(length(c)/2)+1;
% Plot frequency response of yulewalk IIR filter design
plot(frequency_axis(min_f:max_f),amplitude_response(min_f:max_f),'m','LineWidth',linewidth)	% plot interpolated line


% Design a 2nd order Butterwork high-pass filter using "butter" function
[Bb,Ab]=butter(2,(150/(fs/2)),'high');

d=filter(Bb,Ab,a);

frequency_response=fft(d);
amplitude_response=20*log10(abs(frequency_response));
frequency_axis=(0:length(d)-1)*fs/length(d);
min_f=2;
max_f=fix(length(d)/2)+1;
% Plot frequency response of Butterworth IIR filter
plot(frequency_axis(min_f:max_f),amplitude_response(min_f:max_f),'g','LineWidth',linewidth)	% plot interpolated line


e=filter(Bb,Ab,c);

frequency_response=fft(e);
amplitude_response=20*log10(abs(frequency_response));
frequency_axis=(0:length(e)-1)*fs/length(e);
min_f=2;
max_f=fix(length(e)/2)+1;
% Plot combined frequency response of yulewalk and Butterworth IIR filters
plot(frequency_axis(min_f:max_f),amplitude_response(min_f:max_f),'r.','LineWidth',linewidth)	% plot interpolated line


axis([frequency_axis(min_f) frequency_axis(max_f) -90 10])	% Set axis limits
hold off
xlabel('frequency / Hz');
ylabel('amplitude / dB');
grid on	% Draw grid
legend('target','A-weighting','yulewalk','butterworth','combined',3)	% Add legend to graph in bottom left corner

