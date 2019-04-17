function [frames,descriptors,scalespace,difofg]=do_sift(file,varargin)
warning off all;

tic

I=im2double(imread(file)) ;
I=imresize(I,[255,255]);
if(size(I,3) > 1)
  I = rgb2gray( I ) ;
end

[M,N,C] = size(I) ;
% Lowe's choices
S=3 ;
omin= 0 ;
%O=floor(log2(min(M,N)))-omin-4 ; % Up to 16x16 images
O = 4;

sigma0=1.6*2^(1/S) ;
sigmaN=0.5 ;
thresh = 0.2 / S / 2 ; % 0.04 / S / 2 ;
r = 18 ;

NBP = 4 ;
NBO = 8 ;
magnif = 3.0 ;

% Parese input
compute_descriptor = 0 ;
discard_boundary_points = 1 ;
verb = 0 ;

frames = [] ;
descriptors = [] ;

scalespace = do_gaussian(I,sigmaN,O,S,omin,-1,S+1,sigma0) ;


difofg = do_diffofg(scalespace) ;


for o=1:scalespace.O
  
  %  DOG octave
    oframes1 = do_localmax(  difofg.octave{o}, 0.8*thresh, difofg.smin  ) ;
    oframes2 = do_localmax( -difofg.octave{o}, 0.8*thresh, difofg.smin  ) ;
	oframes = [oframes1 ,oframes2 ] ; 
    
    if size(oframes, 2) == 0
        continue;
    end
    
  
    rad = magnif * scalespace.sigma0 * 2.^(oframes(3,:)/scalespace.S) * NBP / 2 ;%
    sel=find(...
      oframes(1,:)-rad >= 1                     & ...
      oframes(1,:)+rad <= size(scalespace.octave{o},2) & ...
      oframes(2,:)-rad >= 1                     & ...
      oframes(2,:)+rad <= size(scalespace.octave{o},1)     ) ;       
      oframes=oframes(:,sel) ;
	
   	oframes = do_extrefine(...
 		oframes, ...
 		difofg.octave{o}, ...
 		difofg.smin, ...
 		thresh, ...
 		r) ;
    
    if size(oframes, 2) == 0
        continue;
    end
   
	oframes = do_orientation(...
		oframes, ...
		scalespace.octave{o}, ...
		scalespace.S, ...
		scalespace.smin, ...
		scalespace.sigma0 ) ;
	
		
  % Store frames
  
	x     = 2^(o-1+scalespace.omin) * oframes(1,:) ;
	y     = 2^(o-1+scalespace.omin) * oframes(2,:) ;
	sigma = 2^(o-1+scalespace.omin) * scalespace.sigma0 * 2.^(oframes(3,:)/scalespace.S) ;	
	frames = [frames, [x(:)' ; y(:)' ; sigma(:)' ; oframes(4,:)] ] ;

	
		
	sh = do_descriptor(scalespace.octave{o}, ...
                    oframes, ...
                    scalespace.sigma0, ...
                    scalespace.S, ...
                    scalespace.smin, ...
                    'Magnif', magnif, ...
                    'NumSpatialBins', NBP, ...
                    'NumOrientBins', NBO) ;
    
    descriptors = [descriptors, sh] ;
    
      
    
end 
fprintf('SIFT features extrating');
toc
fprintf('SIFT: total number of feature points: %d \n\n\n', size(frames,2)) ;
