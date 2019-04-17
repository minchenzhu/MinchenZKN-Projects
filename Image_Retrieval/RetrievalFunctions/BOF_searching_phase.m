function index = BOF_searching_phase(trainedData,ImgPath_search,KMeans)
       
        K=500;
        % load image to search
%         image = imread(ImgPath_search);
        

        % SIFT for image to search
        [~,descr,~,~ ] = do_sift(ImgPath_search, 'Verbosity', 1, 'NumOctaves', 4, 'Threshold',  0.1/3/2 ) ;
        % calculate K-dimensional vector for the image to search
        [cosVector] = get_singleVector(KMeans,K,descr');

        % according to the cosine similarity theorem, find the cosine angle between
        % the image to search and all other images in database
        cosValues = zeros(1,size(trainedData,1));
        for N =1:size(trainedData,1)
            dotprod = sum(cosVector .* trainedData(N,:));
            dis = sqrt(sum(cosVector.^2))*sqrt(sum(trainedData(N,:).^2));
            cosin = dotprod/dis;
            cosValues(N) = cosin;
        end;

        % sort the results
        [~,index] = sort(acos(cosValues));
        index = index(1:10);  
end

