function index =CNN_searching_phase( trainedData,ImgPath_search)

run vl_setupnn
net = load('imagenet-vgg-f') ;
load('PCA_coef');

oriImg = imread( ImgPath_search);
if size(oriImg, 3) == 3
    im_ = single(oriImg) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    res = vl_simplenn(net, im_) ;
    
    featVec = res(20).x;
    featVec = featVec(:);
else
    im_ = single(repmat(oriImg,[1 1 3])) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    res = vl_simplenn(net, im_) ;
    
    featVec = res(20).x;
    featVec = featVec(:);
end

featVec= double(normalize1(featVec));

featVec = featVec'*PCA_coef(:, 1:128);

rows = size(trainedData,1);
test_mat = repmat(featVec,rows,1);
search_compare= sqrt(sum(((test_mat - trainedData).^2)'));

[~,index] = sort(search_compare,'ascend');
index = index(1:10);

end

