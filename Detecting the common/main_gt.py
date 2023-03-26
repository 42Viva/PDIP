import numpy as np
import cv2,os,time

def getSSdescriptor(img,patchsize,regionsize,alpha,orientations,radialDirections):
    '''
    getSSdescriptor用来产生一幅图像的局部自相似描述子（只计算中心像素）
    输入：二维图像矩阵(double型，0 - 1区间)row * col的矩阵，w是矩形框宽，h是矩形框高
    输出：各像素点的局部自相似描述子row * col *（orientations * radialDirections）的矩阵，其中orientations是描述子的角度划分，radialDirections是描述子的角半径划分
    patchsize是计算像素局部自相似描述子时的patch大小
    regionsize是计算像素局部自相似描述子时的region大小
    '''

    row, col = img.shape[:2]
    hps = int(np.round((patchsize-1)/2))
    hrs = int(np.round((regionsize-1)/2))#hrs = half region size

    patch = np.zeros((patchsize,patchsize),dtype = np.float64)
    region = np.zeros((regionsize,regionsize),dtype = np.float64)
    effreg = np.zeros((regionsize - patchsize + 1,regionsize - patchsize + 1),dtype = np.float64)#effective region
    ssdescriptor = np.zeros((row, col, orientations * radialDirections))

    #为了在图像边缘更方便计算，对图像padding
    imgpad = np.pad(img, ((int(hrs), int(hrs)), (int(hrs),int(hrs))),mode = 'constant', constant_values=0)

    #求每一个点的局部自相似描述子
    for pixr in range(1,row+1):
        for pixc in range(1,col+1):
            region = imgpad[pixr - 1 : pixr + 2 * hrs, pixc -1: pixc + 2 * hrs]
            #现在有了region，就可以通过它来计算img(60,100)像素的descriptor了
            patch = region[hrs - hps : hrs + hps + 1, hrs -hps : hrs + hps + 1]
            #以5*5的patch为滑动窗口，去计算和每一个像素邻域的SSD
            #effreg用来存储每一个像素邻域的SSD（平方误差和）
            #范围是region(hps+1:regionsize-hps,hps+1:regionsize-hps)即region(3:59,3:59)

            for i in range(hps+1, regionsize - hps +1):
                for j in range(hps+1, regionsize - hps +1):
                    ssd = np.sum(np.power(region[i - 3:i + 2, j - 3:j + 2] - patch, 2))
                    effreg[i - hps - 1][j - hps - 1] = ssd

            #计算相似性，看一看这个像素点的Correlation Surface
            #alpha = -0.5
            similarity = np.exp(alpha * effreg)

            #下面计算这个像素点的image descriptor 这个descriptor有10个orientations 3个radial directions 共30bins
            #输出一个极坐标矩阵，包含各个像素位置的极坐标信息
            #下面从similarity矩阵里提取当前像素的描述子descriptor

            imageDescriptor = getdescriptor(similarity, orientations, radialDirections)
            # print('第二步用时', time.time() - start_time)
            imageDescriptor_1d = imageDescriptor.flatten()

            ssdescriptor[pixr-1][pixc-1][:] = imageDescriptor_1d
    print('用时', time.time() - start_time)
    return ssdescriptor

def matrixPolarCoordinates(x,y):
    polarCoordinates = np.zeros((2, x, y))
    centerX = np.double((x + 1) / 2)
    centerY = np.double((y + 1) / 2)
    for i in range(1, x+1):
        for j in range(1, y+1):
            dx = np.double(i-centerX)
            dy = np.double(j-centerY)#个像素相对中点位置的笛卡尔坐标
            rho = (dx ** 2 + dx ** 2)/((min(x,y)-1)/2)**2
            theta = np.arctan2(dy,dx)
            polarCoordinates[0][i-1][j-1] = rho
            polarCoordinates[1][i-1][j-1] = theta
    return polarCoordinates


# 把polarCoordinates矩阵对应成各个bin的编号
def numberBins(polarCoordinates,orientations,radialDirections):
    #orientations设置角度区间（超参）,%radialDirections设置半径区间（超参）
    PCdeep, PCrow, PCcol = polarCoordinates.shape
    anglerange = 3.1416*2 / orientations
    distancerange = 1 / radialDirections
    polarBins = np.zeros((2,PCrow,PCcol))

    polarCoordinates[1][:][:] = polarCoordinates[1][:][:] + 3.1416
    #角度的值域实际操作中为(0,2*pi]
    for i in range(1,PCrow+1):
        for j in range(1, PCcol+1):
            for rd in range(1, radialDirections+1):
                if (distancerange * (rd-1)) < polarCoordinates[0][i-1][j-1] and polarCoordinates[0][i-1][j-1]<= (distancerange * rd):#半径区间
                    polarBins[0][i-1][j-1] = rd
            #下面切分角度
            for o in range(1, orientations+1):
                if (o * anglerange - anglerange/2) < polarCoordinates[1][i-1][j-1] and polarCoordinates[1][i-1][j-1] <= (o * anglerange + anglerange/2):#半径区间
                    polarBins[1][i-1][j-1] = o
                if (2*3.1416 - anglerange/2) < polarCoordinates[1][i-1][j-1] or polarCoordinates[1][i-1][j-1] <= (anglerange/2):#半径区间
                    polarBins[1][i-1][j-1] = orientations

    polarBins[0][int((PCrow+1)/2 - 1)][int((PCcol+1)/2 - 1)] = 0 #不考虑中心像素
    return polarBins


def getdescriptor(similarity,orientations,radialDirections):#输入二维数组similarity，输出他的局部描述子
    row, col = similarity.shape
    polarCoordinates = matrixPolarCoordinates(row,col)
    polarBins = np.uint8(numberBins(polarCoordinates,orientations,radialDirections))
    imageDescriptor = np.zeros((radialDirections,orientations))
    for i in range(1,row+1):
        for j in range(1,col+1):
            r = polarBins[0][i-1][j-1]
            o = polarBins[1][i-1][j-1]
            if r>0 and o>0:
                if similarity[i-1][j-1] > imageDescriptor[r-1][o-1]:
                    imageDescriptor[r-1][o-1] = similarity[i-1][j-1]
    return imageDescriptor




patchsize = 5
regionsize = 61
alpha = -0.5
orientations = 10
radialDirections = 3

#读取所有图片
img_path = r'E:\Projects\CVpractice\Exp2\imgs'
imgs = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
n_img = len(imgs)
src_img = []
for i in range(n_img):
    pic = cv2.imread(os.path.join(img_path, imgs[i]))#读入图像
    pic = cv2.resize(pic,(150,100),interpolation=cv2.INTER_CUBIC)#调整大小
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)#转为灰度图
    pic = pic.astype('float64')/255.0 #转为双精度并归一化
    src_img.append(pic)

descriptors = []
for img in src_img:
    start_time = time.time()
    ssdescriptor = getSSdescriptor(img,patchsize,regionsize,alpha,orientations,radialDirections)
    descriptors.append(ssdescriptor)