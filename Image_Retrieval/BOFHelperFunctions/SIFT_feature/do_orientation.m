function oframes = do_orientation(oframes, octave, S, smin,sigma0 )
% this function computes the major orientation of the keypoint (oframes).
% Note that there can be multiple major orientations. In that case, the
% SIFT keys will be duplicated for each major orientation
% Author: Yantao Zheng. Nov 2006.  For Project of CS5240

frames = [];                  
win_factor = 1.5 ;  
NBINS = 36;                   %ֱ��ͼ36��
histo = zeros(1, NBINS);      %��ʼ��ֱ��ͼ����
[M, N, s_num] = size(octave); % M ��ͼ��ĸ߶�, N ��ͼ��Ŀ��; num_level �Ǹ���߶ȿռ�Ĳ���

key_num = size(oframes, 2);        %�ؼ����Ŀ
magnitudes = zeros(M, N, s_num);   %���������������
angles = zeros(M, N, s_num);       %���ط������
% compute image gradients ����ͼ���ݶ�
for si = 1: s_num
    img = octave(:,:,si);
    dx_filter = [-0.5 0 0.5];
    dy_filter = dx_filter';%�Ծ������ת������
    gradient_x = imfilter(img, dx_filter);
    gradient_y = imfilter(img, dy_filter);
    magnitudes(:,:,si) =sqrt( gradient_x.^2 + gradient_y.^2);
    angles(:,:,si) = mod(atan(gradient_y ./ (eps + gradient_x)) + 2*pi, 2*pi);%����Ƕȣ�eps����С����ֵ�Է���ĸΪ�㣬+��*pi�ǽ����Ƕ�ת��Ϊ��ֵ�Ƕ�
end

if size(oframes, 2) == 0
    return;
end
% round off the cooridnates and 
x = oframes(1,:);
y = oframes(2,:) ;
s = oframes(3,:);

x_round = floor(oframes(1,:) + 0.5);%��������
y_round = floor(oframes(2,:) + 0.5);
scales = floor(oframes(3,:) + 0.5) - smin;


for p=1:key_num         %��ÿ���ؼ����д���
    s = scales(p);
    xp= x_round(p);
    yp= y_round(p);
    sigmaw = win_factor * sigma0 * 2^(double (s / S)) ;  %��˹��Ȩ���ӡ�sigma0 * 2^(double (s / S))�ǵ�ǰ��ͼ��ĸ�˹�߶ȣ�
    W = floor(3.0* sigmaw);                              %��˹��Ȩ����ֱ��
    
    for xs = xp - max(W, xp-1): min((N - 2), xp + W)%%xp-1�ǵ�ǰ�㵽�ұ߽�ľ��룬�׾��Ǽ�Ȩ���ڵ���ǰ��ľ��룬
        for ys = yp - max(W, yp-1) : min((M-2), yp + W)
            dx = (xs - x(p));
            dy = (ys - y(p));
            if dx^2 + dy^2 <= W^2 % ���ڸ�˹��ȨԲ��
               wincoef = exp(-(dx^2 + dy^2)/(2*sigmaw^2));
               bin = round( NBINS *  angles(ys, xs, s+ 1)/(2*pi) + 0.5); %���ȡ��ȷ�����ڷ���ֱ��ͼ����
              
               histo(bin) = histo(bin) + wincoef * magnitudes(ys, xs, s+ 1); %�ø�˹��Ȩ������������ֱ��ͼֵ�ۼ�
            end
            
        end
    end
    
    theta_max = max(histo);   %�ҵ�ֱ��ͼ��ֵ
    theta_indx = find(histo> 0.8 * theta_max); %�ؼ�㷽���������80%��ֵ�ĽǶ�
    
    for i = 1: size(theta_indx, 2)
        theta = 2*pi * theta_indx(i) / NBINS;
        frames = [frames, [x(p) y(p) s theta]']; %%%%%%%       
    end   
end

oframes = frames;
% for each keypoint