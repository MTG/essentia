function oldresult = danceability(filename, outfile);
% computes detrended fluctuation analysis exponents.
% wavefile can be mono or stereo (is converted to mono),
% samplingrate can be anything above/equal 11kHz,
% bit depth can be anything above/equal 8 Bit.
% 
% The algorithm is designed after: 
% Jennings et al.; "Variance fluctuations in nonstationary time series:
% a comparative study of music genres"; Condensed Matter, Dec. 2003
% http://arxiv.org/abs/cond-mat/0312380
%
% modifications: 
% - more efficient computation
% - separate consideration of beatinduction and longer term correlation

filename

[samps, fs] = wavread(filename);

if size(samps,2)==2
    % mono downmix
    samps = 0.5*( samps(:,1)+samps(:,2));
end

% preprocessing: 10ms frames
ms10 = round(fs*0.01);
frames = floor(length(samps)/ms10);
clear input;
input(ms10,frames) = 0;
input(:)= samps(1:frames*ms10);
clear samps

% random walk on frame intensity
y = std(input,0,1);
clear input;
y = y - mean(y);
y = cumsum(y);

% DFA computation
tauReal = 310; %initial timescale 31->310ms
maxtau = 10000; %maximal timescale 1000->10s
taufak = 1.1;

j = 1;
while tauReal<=maxtau
    
    tau = floor(tauReal/10);
   
    clear delta_y;
    if(tau<=frames)
        % indexing
        temp = 1:tau;
        jump = floor(tau/50);
        if(jump<1)
            jump = 1;
        end
        n_rows = floor((frames-tau+1)/jump);
        y_sub = jump*ones(tau, n_rows);
        y_sub(:,1) = temp';
        y_sub = cumsum(y_sub,2);
        
        % copy values
        y_sub = y(y_sub);
        
        % detrending
        delta_y = detrend(y_sub);
        
        % average squared error per window
        delta_y = mean(delta_y.^2,1);
        
        %DFA fluctuation
        Fd(j) = sqrt(mean(delta_y));
        
        if j>1
            %DFA exponent alpha
            alpha(j-1) = log10(Fd(j)/Fd(j-1))/log10((tau+3)/(tau_old+3));
        end %if j>1
    else %if(tau<=frames)
        % section too short
        Fd(j) = 0;
        alpha(j-1) = alpha(j-2);
    end
    tau_old = tau;
    tauReal = tauReal * taufak;
    
    j = j+1;
end; % while tau<=maxtau

oldresult = mean(alpha);
result = alpha;

if result(1)~=0
    % valid for tau = 31, taufak = 1.1
    beatmin = find( (result(1:15)>=result(2:16))&(result(3:17)>result(2:16)) ) + 1;
    beatresult = min(result(beatmin));
    if (isempty(beatresult)|(beatresult>1))
        beatresult = 1;
    end
    longresult = mean(result(17:end));
    danceComp = 0.5*beatresult + 0.5*longresult;
else
    danceComp = 0;
    longresult = 0;
    beatresult = 0;
end

% file dump if specified
if nargin>1
    fid = fopen(outfile,'wt');
    fprintf(fid,'%f\n%f\n%f\n%f\n',danceComp, beatresult, longresult, oldresult);
    fclose(fid);
end %if nargin>1
