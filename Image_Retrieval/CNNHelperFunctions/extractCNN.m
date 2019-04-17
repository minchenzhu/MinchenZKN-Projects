
addpath(genpath(pwd));
% run vl_compilenn
run vl_setupnn

%% Step 1 lOADING PATHS
path_imgDB =uigetdir(pwd,'Please select data set folder.');
addpath(path_imgDB);
addpath tools;

net = load('imagenet-vgg-f') ;

%% Step 2 LOADING IMAGE AND EXTRACTING FEATURE

dirOutput = dir(fullfile(path_imgDB,'*.jpg'));
for n = 1:length(dirOutput);
    dirOutput(n).name = [path_imgDB,'/',dirOutput(n).name];
end
imFiles = {dirOutput.name}';

numImg = length(imFiles);
feat = [];
rgbImgList = {};

for i = 1:numel(imFiles)
    i
    imFiles{i}
   oriImg = imread( imFiles{i}); 
   if size(oriImg, 3) == 3
       im_ = single(oriImg) ; % note: 255 range
       im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
       im_ = im_ - net.meta.normalization.averageImage ;
       res = vl_simplenn(net, im_) ;
       
      
       featVec = res(20).x;
       
       featVec = featVec(:);
       feat = [feat; featVec'];
       fprintf('extract %d image\n\n', i);
   else
       im_ = single(repmat(oriImg,[1 1 3])) ; % note: 255 range
       im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
       im_ = im_ - net.meta.normalization.averageImage ;
       res = vl_simplenn(net, im_) ;
       
       featVec = res(20).x;
       
       featVec = featVec(:);
       feat = [feat; featVec'];
       fprintf('extract %d image\n\n', i);
   end
end

% normalization
feat= double(normalize1(feat));

% reduce demension by PCA to 128 dimension.
[coeff, score, latent] = princomp(feat);
CNN = feat;
PCA_coef = coeff;
