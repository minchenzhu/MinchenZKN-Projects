function index =color_searching_phase( trainedData,ImgPath_search)

        Img2search = single(imread(ImgPath_search));
%         ImG = gpuArray(Img2search);
        ImG = Img2search;
        s_search=size(ImG);
        rr=ImG(:,:,1);
        rr=reshape(rr,[s_search(1),s_search(2)]);
        r=mean(mean(rr));
        gg=ImG(:,:,2);
        gg=reshape(gg,[s_search(1),s_search(2)]);
        g=mean(mean(gg));
        bb=ImG(:,:,3);
        bb=reshape(bb,[s_search(1),s_search(2)]);
        b=mean(mean(bb));

        serach_data = repmat([r,g,b],size(trainedData,1),1);

        search_compare = sqrt((trainedData(:,1) - serach_data(:,1)).^2 + ...
            (trainedData(:,2) - serach_data(:,2)).^2 + ...
            (trainedData(:,3) - serach_data(:,3)).^2);

        [~,index] = sort(search_compare,'ascend');
        index = index(1:10);
        
end

