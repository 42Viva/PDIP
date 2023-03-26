
%getSSdescriptor用来产生一幅图像的局部自相似描述子（只计算中心像素）
%输入：二维图像矩阵(double型，0-1区间)row*col的矩阵，w是矩形框宽，h是矩形框高                                      
%输出：各像素点的局部自相似描述子row*col*（orientations*radialDirections）的矩阵，其中orientations是描述子的角度划分，radialDirections是描述子的角半径划分
%patchsize是计算像素局部自相似描述子时的patch大小
%regionsize是计算像素局部自相似描述子时的region大小
function ssdescriptor = getSSdescriptor(img,patchsize,regionsize,alpha,orientations,radialDirections)
    %建议patchsize=5，regionsize=61,alpha=-0.5用来调节相似性的计算,orientations=10,radialDirections=3
    [row,col]=size(img);
    hps = round(patchsize-1)/2; hrs= round(regionsize-1)/2;%hrs=halfregionsize
    patch = zeros(patchsize,patchsize,"double");
    region = zeros(regionsize,regionsize,"double");
    effreg = zeros(regionsize-patchsize+1,regionsize-patchsize+1,"double");%effectiveregion
    ssdescriptor = zeros(row,col,orientations*radialDirections);
    %为了在图像边缘处更方便的计算，对矩阵做padding
    imgpad = padarray(img,[hrs hrs],0,'both');
    %后边的操作对象是imgpad矩阵，它的有效部分是imgpad((1+round(regionsize/2)):row+ round(regionsize/2) , (1+round(regionsize/2):col+round(regionsize/2))
    %如果要计算原图img(60,100)这个像素的descriptor
    %但在imgpad中img(60,100)对应imgpad(90,130)也就是imgpad(60+hrs,100+hrs) 
    %下面求每一个点的局部自相似描述子
    %pixr = 1; pixc= 1 ;%这个pixr和pixc是我们需要遍历整张图的一个变量
    for pixr = 1:row
        for pixc = 1:col
            region = imgpad(pixr:pixr+2*hrs,pixc:pixc+2*hrs);%注意我们选的范围
            %现在有了region，就可以通过它来计算img(60,100)像素的descriptor了
            patch = region(hrs+1-hps:hrs+1+hps,hrs+1-hps:hrs+1+hps);%对patch赋region的中心值
            %以5*5的patch为滑动窗口，去计算和每一个像素邻域的SSD
            %effreg用来存储每一个像素邻域的SSD（平方误差和）
            %范围是region(hps+1:regionsize-hps,hps+1:regionsize-hps)即region(3:59,3:59)
            for i = hps+1:regionsize-hps
                for j = hps+1:regionsize-hps
                  ssd = sum(sum(region(i-2:i+2,j-2:j+2)-patch)).^2;%计算平方误差和
                  effreg(i-hps,j-hps) = ssd;
                end
            end
            %%计算相似性，看一看这个像素点的Correlation Surface
            %alpha = -0.5;%这个参数与相似度有关 很重要
            similarity = exp(alpha*effreg) ;
            %下面计算这个像素点的image descriptor 这个descriptor有10个orientations 3个radial directions 共30bins
            %输出一个极坐标矩阵，包含各个像素位置的极坐标信息
            %下面从similarity矩阵里提取当前像素的描述子descriptor
            imageDescriptor = getdescriptor(similarity,orientations,radialDirections);
            imageDescriptor_1d=imageDescriptor(:);%描述子用一维向量表示，是一个(orientations*radialDirections)*1的向量

            ssdescriptor(pixr,pixc,:) = imageDescriptor_1d;
        end
    end

end

%调用的自定义函数
function polarCoordinates = matrixPolarCoordinates(x,y)
    polarCoordinates = zeros(x,y,2);
    centerX = double((x+1)/2);
    centerY = double((y+1)/2);
    for i = 1:x
        for j = 1:y
                dx = double(i-centerX) ;dy = double(j-centerY);%各像素相对中点位置的笛卡尔坐标
                rho = (dx^2+dy^2)/ ((min(x,y)-1)/2)^2;%我们把rho做一个归一化的处理
                theta = atan2(dy,dx);
                polarCoordinates(i,j,1) = rho; polarCoordinates(i,j,2) = theta;
        end
    end
end

%%把polarCoordinates矩阵对应成各个bin的编号
%输入：待给各个bin编号的极坐标矩阵 输出：各个像素位于哪一个bin
%规则：里圈编号小 0°所在位置为1号bin  角度区间(-3.1416,3.1416]
function polarBins =  numberBins(polarCoordinates,orientations,radialDirections)
    %orientations设置角度区间（超参）,%radialDirections设置半径区间（超参）
    [PCrow,PCcol,PCdeep] = size(polarCoordinates);%★★在这里一定要加上PCdeep
    anglerange = 3.1416*2/orientations;%每一个扇形区域的角度
    distancerange = 1/radialDirections;
    polarBins = zeros(PCrow,PCcol,2);

    polarCoordinates(:,:,2) = polarCoordinates(:,:,2)+3.1416;%为了方便分割角度★★这句话一定要在循环的外面
    % 角度的值域实际操作中为(0,2*pi]
    for i=1:PCrow
        for j=1:PCcol
            for rd = 1:radialDirections
                if ( ((distancerange*(rd-1)) < polarCoordinates(i,j,1))&&(polarCoordinates(i,j,1)<= (distancerange*rd)) )%半径区间
                    polarBins(i,j,1) = rd;              
                end
            end
                %下面是对角度做的切分
                
            for o = 1:orientations
                if ( (o*anglerange-anglerange/2) < polarCoordinates(i,j,2) )&&( polarCoordinates(i,j,2) <= (o*anglerange+anglerange/2) )%半径区间
                    polarBins(i,j,2) = o; 
                end
            end
                if ( ((2*3.1416-anglerange/2) < polarCoordinates(i,j,2) ) || ( polarCoordinates(i,j,2) <= (anglerange/2)) )
                    polarBins(i,j,2) = orientations;
                end
        end
    end

    polarBins((PCrow+1)/2,(PCcol+1)/2,1) = 0;%中心像素不在我们的考虑范围之内，距离为0

end

%输入二维数组similarity，输出他的局部描述子
function imageDescriptor =  getdescriptor(similarity,orientations,radialDirections)
    [row,col] = size(similarity);
    polarCoordinates = matrixPolarCoordinates(row,col);
    polarBins =  uint8(numberBins(polarCoordinates,orientations,radialDirections));%区间划分 polarBins:row*col*2
    imageDescriptor = zeros(radialDirections,orientations);%3*10=30bins
    for i = 1:row
        for j =1:col
            %选每一个bin中的最大值
            r = polarBins(i,j,1); o = (polarBins(i,j,2));   
            if ((r>0) && (o>0)) %防止索引为0
               if ( similarity(i,j) > imageDescriptor(r,o) )%更新最大值
                    imageDescriptor(r,o) = similarity(i,j);
               end
            end
        end
    end
end