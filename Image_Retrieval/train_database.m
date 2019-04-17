warning ('off','all'); 
addpath(genpath(pwd));

%% read in the images of training set
 ImgPath=uigetdir(pwd,'Please select data set folder.');
 ImgFormat = '*.jpg';
 dirOutput = dir(fullfile(ImgPath,ImgFormat));
 for n = 1:length(dirOutput);
     dirOutput(n).name = [ImgPath,'/',dirOutput(n).name];
 end
 imFiles = {dirOutput.name}';

 %% Method: avg. color value of 3 channels
 AvgC  = color_training_phase( imFiles );
 save('AvgC.mat',AvgC);
 
 %% Method: color moments
cMom = cMom_training_phase( imFiles);
save('cMom.mat',cMom);

%% Method: LBP
LBP = LBP_training_phase( imFiles );
save('LBP.mat',LBP);
 
%% Method: BOF
[ BOF,KMeans ] = BOF_training_phase( imFiles);
save('BOF.mat',BOF);
save('clustering.mat',KMeans);

%% Method: CNN
 [CNN,PCA_coef ] = CNN_training_phase(imFiles);
 save('CNN.mat',CNN);
 save('PCA_coef.mat',PCA_coef);
