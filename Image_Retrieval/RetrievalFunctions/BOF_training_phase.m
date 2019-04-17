function [ BOF,KMeans ] = BOF_training_phase( imFiles)
    
    % number of clusters
    K=500;

    % SIFT for images in database
    % [img_paths,Feats] = get_sifts_2(imFiles);
    [Feats] = get_sifts_2(imFiles);

    % generate K means cluster centers
    initMeans = Feats(randi(size(Feats,1),1,K),:);

    % clustering
    [KMeans] = K_Means(Feats,K,initMeans);

    % count the number of feature points in each cluster
    % every image has a K-dimensional vector
    BOF = get_countVectors(KMeans,K,size(imFiles,1));

end

