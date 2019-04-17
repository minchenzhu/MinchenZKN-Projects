function cMom = cMom_training_phase( imFiles)
       
        cMom = zeros(225,numel(imFiles));
        
        w = waitbar(0,'Now training...','facecolor','g');
        for i=1:numel(imFiles)
            img2train = imread(imFiles{i});
            cMom(:,i)=colorMom(img2train);
             str=['Now training...',num2str(i),'/',num2str(numel(imFiles))];
            waitbar(i / numel(imFiles),w,str,'facecolor','g');
        end
        cMom = cMom';
        close(w);
        
end

