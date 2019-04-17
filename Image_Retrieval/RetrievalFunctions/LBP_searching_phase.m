function index = LBP_searching_phase(trainedData,ImgPath_search)

    img= imread(ImgPath_search);
    LBPVec=lbp(img);
    
    rows = size(trainedData,1);
    test_mat = repmat(LBPVec,rows,1);
    search_compare= sqrt(sum(((test_mat - trainedData).^2)'));
    
    [~,index] = sort(search_compare,'ascend');
    index = index(1:10);

end

