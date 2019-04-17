warning ('off','all'); 
addpath(genpath(pwd));

stepSize = 1;
log2c_list = -5:stepSize:5;
log2g_list = -5:stepSize:5;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
cvMatrix = zeros(numLog2c,numLog2g);
bestcv = 0;

% load('label.mat');
load('label_test_1');
label = label_test_1;
% load('Color256.mat');
% trainData = trainedData;
% load('BOF256.mat');
% trainData = countVectors;
load('BOF_test_1.mat');
trainData = BOF_test_1;

%%
tic
for i = 1:numLog2c
    log2c = log2c_list(i);
    for j = 1:numLog2g
        log2g = log2g_list(j);
        % -v 3 --> 3-fold cross validation
        param = ['-q -v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(label, inst, param);
        cvMatrix(i,j) = cv;
        if (cv >= bestcv),
            bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end
toc
stepSize = 5;
log2c_list = -50:stepSize:50;
log2g_list = -50:stepSize:50;
numLog2c = length(log2c_list);
numLog2g = length(log2g_list);

figure;
imagesc(crossValidation11); colormap('jet'); colorbar;
set(gca,'XTick',1:10)
set(gca,'XTickLabel',sprintf('%d|',log2g_list))
xlabel('Log_2\gamma');
set(gca,'YTick',1:10)
set(gca,'YTickLabel',sprintf('%d|',log2c_list))
ylabel('Log_2c');