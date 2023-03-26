import cv2,os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


img_path = r'E:\Projects\CVpractice\Exp2\imgs'
imgs = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
n_img = len(imgs)
thumbs = []
for i in range(n_img):
    thumbs.append(cv2.imread(os.path.join(img_path, imgs[i])))
#
# r=[]
# a=[]
# for img in thumbs:
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = gray / 255.0 #像素值0-1之间
#
#     #sobel算子分别求出gx，gy
#     gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)#x方向的导数
#     gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)#y方向的导数
#     """直角坐标系转换为极坐标系"""
#     mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=1) #得到梯度幅度和梯度角度阵列
#     r.append(mag)
#     a.append(ang)
#
# for i in r:
#     cv2.imshow('gray',i)
#     cv2.waitKey(0)

def cart2polar(region_size, num_bins):
    radius = np.zeros(region_size)
    angle = np.zeros(region_size)
    center = (region_size[0] // 2, region_size[1] // 2)
    for row in range(region_size[0]):
        for col in range(region_size[1]):
            # 计算当前像素点的极坐标半径和角度
            theta, rho = np.arctan2(row - center[0], col - center[1]), np.sqrt((row - center[0])**2 + (col - center[1])**2)
            # 将极坐标角度从弧度转换为度数，并将角度映射到[0, 360)区间
            angle[row, col] = (theta * 180 / np.pi) % 360
            # 将极坐标半径量化到3个径向间隔bin中的一个，量化方法为对数变换
            log_rho = np.log(rho)
            rho_bins = np.linspace(0, np.max(log_rho), num=num_bins+1)
            for i in range(num_bins):
                if rho_bins[i] <= log_rho < rho_bins[i+1]:
                    radius[row, col] = i
                    break
    return radius, angle


def com_self_similarities(src_image, region_size, patch_size, bin):
    # 转换到 Lab 颜色空间
    lab_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB)

    # 计算参数
    vec_size = 45 #num_bins
    alpha = 1 / (85**2)
    center_region = (region_size[0]//2, region_size[1]//2)
    center_patch = (patch_size[0]//2, patch_size[1]//2)
    lab_size = lab_image.shape

    # 初始化数组
    self_similarities = np.zeros((lab_size[0], lab_size[1], vec_size))

    # 计算每个像素点的自相似性描述子
    for row in range(center_region[0]+1, lab_size[0]-center_region[0]):
        for col in range(center_region[1]+1, lab_size[1]-center_region[1]):
            patch = lab_image[row-center_patch[0]:row+center_patch[0]+1, col-center_patch[1]:col+center_patch[1]+1, :]#划定patch
            region = lab_image[row-center_region[0]:row+center_region[0]+1, col-center_region[1]:col+center_region[1]+1, :]#划定region
            ssd_region = cal_ssd(patch, region, alpha, center_patch)#得到patch和region的相似性矩阵
            vec = get_self_sim_vec(ssd_region, bin, vec_size)#相似性转换为自相似性描述子
            LSSD, ps = cv2.normalize(vec.reshape(-1, 1), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F, None)# 对描述子归一化
            self_similarities[row, col, :] = LSSD.flatten()

    return self_similarities


def cal_ssd(patch, region, alpha, center_patch):
    # 计算相似度
    patch = patch.astype(np.float32)
    region = region.astype(np.float32)
    diff = patch - region[center_patch[0], center_patch[1], :]
    ssd_region = np.sum(diff**2, axis=2)
    ssd_region = np.exp(-alpha*ssd_region)

    return ssd_region


def get_self_sim_vec(ssd_region, bin, vec_size):
    # 将相似度转换为自相似描述子
    # self_sim_vec = np.zeros(vec_size)
    # for m in range(14):
    #     theta_low = m*24
    #     theta_up = (m+1)*24
    #     for n in range(3):
    #         rho_low = bin[m][n][0]
    #         rho_up = bin[m][n][-1]
    #
    #         temp = np.zeros_like(ssd_region)
    #         for i in range(len(bin[m][n])):
    #             temp[bin[m][n][i][0], bin[m][n][i][1]] = ssd_region[bin[m][n][i][0], bin[m][n][i][1]]
    #
    #         if np.sum(temp) > 0:
    #             self_sim_vec[m*3+n] = np.sum(temp) / (np.sum(temp**2)**0.5)
    #
    # return self_sim_vec
    hist, _ = np.histogram(ssd_region, bins=bin, range=(0, np.pi), density=False)
    hist = hist / np.sum(hist)
    ssd_vec = np.zeros((1, vec_size))
    for i in range(bin):
        ssd_vec[0, i::bin] = hist[i]
    return ssd_vec

region = [60,60]
patch = [5,5]
bin = 80
num_bins = 45
# radius, angle = cart2polar(region,num_bins)
# cv2.imshow("mag", radius)
# cv2.imshow(" ang",  angle)
# cv2.waitKey(0)
# print(angle)
s = com_self_similarities(thumbs[0],region,patch,bin)
# print(s)