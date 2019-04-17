function [distance] = do_eucidean_distance( object1, object2 )
% DO_EUCIDEAN_DISTANCE  用于求两个多维向量的欧氏距离
% WRITTEN BY GUOCHUAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT  --两个高维向量
% OUTPUT --欧氏距离

results = (object1-object2).^2;
distance = sum(results);


