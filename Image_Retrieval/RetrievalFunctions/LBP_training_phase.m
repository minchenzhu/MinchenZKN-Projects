function LBP = LBP_training_phase( imFiles )

       LBP = zeros(256,numel(imFiles));
        
        w = waitbar(0,'Now training...','facecolor','g');
        for i=1:numel(imFiles)
            img2train = imread(imFiles{i});
            LBP(:,i)=lbp(img2train);
           str=['Now training...',num2str(i),'/',num2str(numel(imFiles))];
            waitbar(i / numel(imFiles),w,str,'facecolor','g');
        end
        LBP = LBP';
        close(w);

end

