% Compute high-resolution tonal descriptors (HPCP)
% for a collection of files

% Folder storing western music descriptors
path_w='E:\Data\western_vs_non_western_44K_mono\Descriptors\Evaluation-April2007-AfterTuning\western\';
%path_w='E:\Data\western_vs_non_western_44K_mono\Descriptors\Evaluation-April2007-AfterTuning\classical\';

% Folder storing non-western music descriptors
path_nw='E:\Data\western_vs_non_western_44K_mono\Descriptors\Evaluation-April2007-AfterTuning\non_western\';

global et_dev nt2t_energyratio nt2t_peaksenergyratio nw_r;

western=0;
showplots=1;

if western
    path=path_w;
else
    path=path_nw;
end
      
filelist=dir([path '*.HPCP.txt']); 
nFiles = length(filelist)

et_dev = [];
nt2t_energyratio = [];
nt2t_peaksenergyratio = [];
nw_r = [];

if western
    fp = fopen('western-30-04-equal_tempered_normalized.txt','w'); 
else
    fp = fopen('non-western-30-04-equal_tempered_normalized.txt','w'); 
end

for i=1:nFiles
    i
    filename=filelist(i).name
    hpcp_frame = load([path filename]);
    hpcpsize = min(size(hpcp_frame));
    nFrames = max(size(hpcp_frame));
    % COMPUTE HPCP120 AVERAGE
    hpcp_av = hpcp_frame(1,:);
    for i=2:nFrames
        hpcp_av = hpcp_av + hpcp_frame(i,:);
    end
    hpcp_av=hpcp_av/nFrames;
    % COMPUTE HPCP120 DERIVED DESCRIPTORS
    % equal_tempered_dev, nontempered2tempered_energy_ratio
    % and nontempered2tempered_peaks_energy_ratio
     [equal_tempered_dev nontempered2tempered_energy_ratio nontempered2tempered_peaks_energy_ratio]=ComputeHighResolutionFeatures(hpcp_av');
    et_dev = [et_dev equal_tempered_dev];
    nt2t_energyratio = [nt2t_energyratio nontempered2tempered_energy_ratio];
    nt2t_peaksenergyratio = [nt2t_peaksenergyratio nontempered2tempered_peaks_energy_ratio];
    % READ KEY STRENGTH
    keyfilename=strrep(filename,'HPCP','key');
    [keynote, mode, strength] = textread([path keyfilename],'%s\t%s\t%f',1);
    tuning = textread([path keyfilename], 'tuning %f', 1, 'headerlines', 1);
    if (strength>0)
        nw_r = [nw_r; strength];
    end

    fprintf(fp,'%s\t%f\t%f\t%f\t%f\t%f western\n',filename,strength,tuning,equal_tempered_dev,nontempered2tempered_energy_ratio,nontempered2tempered_peaks_energy_ratio); 

end

fclose(fp);
[corrhist,x] = hist(et_dev);
[corrhist2,x2] = hist(nt2t_energyratio);
[corrhist3,x3] = hist(nt2t_peaksenergyratio);
[corrhist4,x4] = hist(nw_r);

if showplots ==1
    if western==0
        figure(3)
        subplot(212);
        bar(x,corrhist./max(corrhist));
        grid;
        title('Histogram of equal-tempered deviation for non-western music');
        figure(4)
        subplot(212);
        bar(x2,corrhist2./max(corrhist2));
        grid;
        title('Histogram of non-tempered 2 tempered energy ratio for non-western music');
        figure(5)
        subplot(212);
        bar(x4,corrhist4./max(corrhist4));
        grid;
        title('Histogram of max correlation with diatonic profile for non-western music');
        figure(6)
        subplot(212);
        bar(x3,corrhist3./max(corrhist3));
        grid;
        title('Histogram of non-tempered 2 tempered peaks energy ratio for non-western music');
    else
        figure(3)
        subplot(211);
        bar(x,corrhist./max(corrhist));
        grid;
        title('Histogram of equal-tempered deviation for western music');
        figure(4)
        subplot(211);
        bar(x2,corrhist2./max(corrhist2));
        grid;
        title('Histogram of non-tempered 2 tempered energy ratio for western music');
        figure(5)
        subplot(211);
        bar(x4,corrhist4./max(corrhist4));
        grid;
        title('Histogram of max correlation with diatonic profile for western music');
        figure(6)
        subplot(211);
        bar(x3,corrhist3./max(corrhist3));
        grid;
        title('Histogram of non-tempered 2 tempered peaks energy ratio for western music');
    end

end

    


