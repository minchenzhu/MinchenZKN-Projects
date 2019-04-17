BOF10= normalize1(BOF_test_1(1:50,:));
TXT10 = normalize1(LBP_txture_test_1(:,1:50)');
COL10 = normalize1(LBP_color_test_1(:,1:50)');
color10 = normalize1(color_test_1(1:50,:));

BOF20 = normalize1(BOF_test_1(1:100,:));
TXT20 = normalize1(LBP_txture_test_1(:,1:100)');
COL20 = normalize1(LBP_color_test_1(:,1:100)');
color20 = normalize1(color_test_1(1:100,:));

BOF30 = normalize1(BOF_test_1(1:150,:));
TXT30 = normalize1(LBP_txture_test_1(:,1:150)');
COL30 = normalize1(LBP_color_test_1(:,1:150)');
color30 = normalize1(color_test_1(1:150,:));


label10 = label_test_1(1:50);
label20 = label_test_1(1:100);
label30 = label_test_1(1:150);




Cs=[10.^(-7:7)];
gammas=[10.^(-7:7)];
ds=1:30;


crossValidation11=[];
resultStruct11=[];
iterator11=1;
crossValidation12=[];
resultStruct12=[];
iterator12=1;
crossValidation13=[];
resultStruct13=[];
iterator13=1;

crossValidation23=[];
resultStruct23=[];
iterator23=1;
crossValidation21=[];
resultStruct21=[];
iterator21=1;
crossValidation22=[];
resultStruct22=[];
iterator22=1;

crossValidation31=[];
resultStruct31=[];
iterator31=1;
crossValidation32=[];
resultStruct32=[];
iterator32=1;
crossValidation33=[];
resultStruct33=[];
iterator33=1;

crossValidation41=[];
resultStruct41=[];
iterator41=1;
crossValidation42=[];
resultStruct42=[];
iterator42=1;
crossValidation43=[];
resultStruct43=[];
iterator43=1;






%% RBF
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,BOF10, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation11(i,j)=cv;
        
        resultStruct11(iterator11).cv=cv;
        resultStruct11(iterator11).gamma=gammas(j);
        resultStruct11(iterator11).C=Cs(i);
        resultStruct11(iterator11).t=2;
        iterator11=iterator11+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label20,BOF20, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation12(i,j)=cv;
        
        resultStruct12(iterator12).cv=cv;
        resultStruct12(iterator12).gamma=gammas(j);
        resultStruct12(iterator12).C=Cs(i);
        resultStruct12(iterator12).t=2;
        iterator12=iterator12+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label30,BOF30, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation13(i,j)=cv;
        
        resultStruct13(iterator13).cv=cv;
        resultStruct13(iterator13).gamma=gammas(j);
        resultStruct13(iterator13).C=Cs(i);
        resultStruct13(iterator13).t=2;
        iterator13=iterator13+1;
        
    end
end

for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,color10, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation21(i,j)=cv;
        
        resultStruct21(iterator21).cv=cv;
        resultStruct21(iterator21).gamma=gammas(j);
        resultStruct21(iterator21).C=Cs(i);
        resultStruct21(iterator21).t=2;
        iterator21=iterator21+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label20,color20, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation22(i,j)=cv;
        
        resultStruct22(iterator22).cv=cv;
        resultStruct22(iterator22).gamma=gammas(j);
        resultStruct22(iterator22).C=Cs(i);
        resultStruct22(iterator22).t=2;
        iterator22=iterator22+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label30,color30, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation23(i,j)=cv;
        
        resultStruct23(iterator23).cv=cv;
        resultStruct23(iterator23).gamma=gammas(j);
        resultStruct23(iterator23).C=Cs(i);
        resultStruct23(iterator23).t=2;
        iterator23=iterator23+1;
        
    end
end

for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,TXT10, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation31(i,j)=cv;
        
        resultStruct31(iterator31).cv=cv;
        resultStruct31(iterator31).gamma=gammas(j);
        resultStruct31(iterator31).C=Cs(i);
        resultStruct31(iterator31).t=2;
        iterator31=iterator31+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label20,TXT20, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation32(i,j)=cv;
        
        resultStruct32(iterator32).cv=cv;
        resultStruct32(iterator32).gamma=gammas(j);
        resultStruct32(iterator32).C=Cs(i);
        resultStruct32(iterator32).t=2;
        iterator32=iterator32+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label30,TXT30, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation33(i,j)=cv;
        
        resultStruct33(iterator33).cv=cv;
        resultStruct33(iterator33).gamma=gammas(j);
        resultStruct33(iterator33).C=Cs(i);
        resultStruct33(iterator33).t=2;
        iterator33=iterator33+1;
        
    end
end

for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,COL10, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation41(i,j)=cv;
        
        resultStruct41(iterator41).cv=cv;
        resultStruct41(iterator41).gamma=gammas(j);
        resultStruct41(iterator41).C=Cs(i);
        resultStruct41(iterator41).t=2;
        iterator41=iterator41+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label20,COL20, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation42(i,j)=cv;
        
        resultStruct42(iterator42).cv=cv;
        resultStruct42(iterator42).gamma=gammas(j);
        resultStruct42(iterator42).C=Cs(i);
        resultStruct42(iterator42).t=2;
        iterator42=iterator42+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label30,COL30, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation43(i,j)=cv;
        
        resultStruct43(iterator43).cv=cv;
        resultStruct43(iterator43).gamma=gammas(j);
        resultStruct43(iterator43).C=Cs(i);
        resultStruct43(iterator43).t=2;
        iterator43=iterator43+1;
        
    end
end

figure,
imagesc(crossValidation52),title('grid search for CNN128 features with 10 categories'); 
colormap('jet'); colorbar;
xlabel('Lg(\gamma)');
set(gca,'XTick',1:15),
set(gca,'XTickLabel',{'-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7'}),
ylabel('Lg(c)'),
set(gca,'YTick',1:15),
set(gca,'YTickLabel',{'-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7'});



CNN_4096 = double(CNN_test_1(1:50,:));
CNN_128 = CNN_test_1_128(1:50,:);

crossValidation51=[];
resultStruct51=[];
iterator51=1;
crossValidation52=[];
resultStruct452=[];
iterator52=1;
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,CNN_4096, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation51(i,j)=cv;
        
        resultStruct51(iterator51).cv=cv;
        resultStruct51(iterator51).gamma=gammas(j);
        resultStruct51(iterator51).C=Cs(i);
        resultStruct51(iterator51).t=2;
        iterator51=iterator51+1;
        
    end
end
for i=1:length(Cs)
    for j=1:length(gammas)
        Cs(i)
        gammas(j)
        cv=svmtrain(label10,CNN_128, ['-s 0 -t 2 -c ' num2str(Cs(i)) ' -gamma ' num2str(gammas(j)) ' -v 2 -q']);
        crossValidation52(i,j)=cv;
        
        resultStruct52(iterator52).cv=cv;
        resultStruct52(iterator52).gamma=gammas(j);
        resultStruct52(iterator52).C=Cs(i);
        resultStruct52(iterator52).t=2;
        iterator52=iterator52+1;
        
    end
end