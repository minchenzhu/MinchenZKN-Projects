


% test = color_test_1;
% train = AvgC;
% test = LBP_test_1';
% train = LBP';
% test = cMom_test_1';
% train = cMom';
% test = CNN_test_1_128;
% train = CNN;

num = size(test,1);
rows = size(train,1);
correct = zeros(1,num);
for i = 1:num
    test_mat = repmat(test(i,:),rows,1);
    EDist = sqrt(sum(((test_mat - train).^2)'));
    if min(EDist) <= 1e-10
        correct(i) = 1;
    else
        correct(i) = 0;
    end
end

correct_accuracy = sum(correct)/num;
sprintf('%2.2f%%', correct_accuracy*100) 


%%
label = label_test_1;
% test = BOF_test_2;
% train = BOF;
% test = color_test_2;
% train = AvgC;
% test = LBP_test_2';
% train = LBP';
% test = cMom_test_2';
% train = cMom';
test = CNN_test_2_128;
train = CNN;
num = size(test,1);
rows = size(train,1);
correct = zeros(1,num);
max_simil = zeros(1,num);
for i = 1:num
    
    low = 75*(label(i)-1)+1;
    up = 75*label(i);
    test_mat = repmat(test(i,:),rows,1);
    numerator = sum((test_mat .*train)');
    denominator = sqrt(sum((test_mat.^2)')).*sqrt(sum((train.^2)'));
    similarity = numerator./denominator ;
    class = find(similarity == max(similarity));
    
    if  class >= low & class <= up
        correct(i) = 1;
    else
        correct(i) = 0;
    end
end

correct_accuracy = sum(correct)/num;
sprintf('%2.2f%%', correct_accuracy*100) 
