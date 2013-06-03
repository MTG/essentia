% Read key strength information and compute an histogram for a set of files
% stored in the same folder.
% These text files are the output of SMSToolsHPCP

% Folder storing western music descriptors
%path_w = 'E:\Sounds\Scales\Test\western\44kmono\';
% path_w = 'E:\Sounds\western_vs_non_western_44K_mono\western\44Kmono_first30seconds\';
path_w = 'E:\Sounds\western_vs_non_western_44K_mono\classical\44Kmono_first30seconds\';    % classical DB
% Folder storing non-western music descriptors
path_nw = 'E:\Sounds\western_vs_non_western_44K_mono\non_western\44Kmono_first30seconds\';

western=1;

if western
    path=path_w;
else
    path=path_nw;
end

filelist=dir([path '*.key.txt']);
nFiles = length(filelist);
nw_r =[];

for i=1:nFiles
    i
    filename=filelist(i).name
    [keynote, mode, strength] = textread([path filename],'%s\t%s\t%f',1)
    if (strength>0)
        nw_r = [nw_r; strength];
    end
end

[corrhist,x] = HIST(nw_r);


if western==0
    figure(5)
    subplot(212);
    bar(x,corrhist./max(corrhist));
    grid;
    title('Histogram of max correlation with diatonic profile for non-western music');
else
    figure(5)
 %   subplot(211);
    bar(x,corrhist./max(corrhist));
    grid;
    title('Histogram of max correlation with diatonic profile for western music');
end

    
