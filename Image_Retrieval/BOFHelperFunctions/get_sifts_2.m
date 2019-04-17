function [Feats] = get_sifts_2( FullFilePaths )


Feats = [];
 w = waitbar(0,'Extracting SIFT Features...','facecolor','g');
for N = 1:numel(FullFilePaths)
        FullFilePaths{N};
        [~,descr,~,~ ] = do_sift( FullFilePaths{N}, 'Verbosity', 1, 'NumOctaves', 4, 'Threshold',  0.1/3/2 ) ; %0.04/3/2
        descr = descr';
        feat_count = size(descr,1);
        descr = [descr,ones(feat_count,1)*N];
        Feats=[Feats;descr];
        str=['Extracting SIFT Features...',num2str(N),'/',num2str(numel(FullFilePaths))];
        waitbar(N / numel(FullFilePaths),w,str,'facecolor','g');
end;
        close(w);
end

