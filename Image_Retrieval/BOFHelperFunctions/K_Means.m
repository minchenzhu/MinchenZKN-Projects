function [ KMeans ] = K_Means( Feats, K , initMeans )


for n = 1:K; 
KMeans(n).value = initMeans(n,1:128); 
KMeans(n).data = initMeans(n,:);
KMeans(n).count = 1;
end;
 w = waitbar(0,'KMeans clustring...');
for N=1:size(Feats,1)
    min = do_eucidean_distance(Feats(N,(1:128)),KMeans(1).value);
    num = 1;
    for M=2:K
        distance = do_eucidean_distance(Feats(N,(1:128)),KMeans(M).value);
        if(distance<min)
            min = distance;
            num = M;
        end;
    end;
    KMeans(num).data = [KMeans(num).data;Feats(N,:)];
    KMeans(num).value = KMeans(num).value * KMeans(num).count+ Feats(N,1:128);
    KMeans(num).count = KMeans(num).count+1;
    KMeans(num).value = KMeans(num).value / KMeans(num).count; 
    str=['Now clustering......',num2str(N)];
    waitbar(N / size(Feats,1),w,str);
end;
    close(w);
    
end

