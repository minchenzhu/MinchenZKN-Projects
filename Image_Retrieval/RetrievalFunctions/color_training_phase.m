function trainedData  = color_training_phase( imFiles )
        addpath(genpath(pwd));
      %% training phase
        trainedData = zeros(numel(imFiles),3);
        w = waitbar(0,'Now training...','facecolor','g');
       
        for i = 1:numel(imFiles)
            img2train = imread(imFiles{i});
            s=size(img2train);
            if length(s) < 3
                img2train = cat(3,img2train,img2train,img2train);
            end
            rr=img2train(:,:,1);
            rr=reshape(rr,[s(1),s(2)]);
            trainedData(i,1)=mean(mean(rr));
            gg=img2train(:,:,2);
            gg=reshape(gg,[s(1),s(2)]);
            trainedData(i,2)=mean(mean(gg));
            bb=img2train(:,:,3);
            bb=reshape(bb,[s(1),s(2)]);
            trainedData(i,3)=mean(mean(bb));
            
            str=['Now training...',num2str(i),'/',num2str(numel(imFiles))];
            waitbar(i / numel(imFiles),w,str,'facecolor','g');
            
    
        end
        close(w);
end

