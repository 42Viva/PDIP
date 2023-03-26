import cv2,os
import numpy as np
import draw_result,com_Self_Similarities,get_bin


#读取所有图片
img_path = r'E:\Projects\CVpractice\Exp2\imgs'
imgs = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
n_img = len(imgs)
thumbs = []
for i in range(n_img):
    thumbs.append(cv2.imread(os.path.join(img_path, imgs[i])))

#转化为极坐标
def cart2polar(region_size):
    w, l = region_size[0], region_size[1]
    radius = np.zeros((w, l),dtype=np.float64)
    angle = np.zeros((w, l),dtype=np.float64)
    center = (np.ceil(w/2),np.ceil(l/2))
    print(center,center.shape)

    for row in range(1,w+1):
        for col in range(1, l+1):
            rhos, thetas = cv2.cartToPolar(row-center[0],col-center[1])
            rho = rhos[0][0]
            theta = thetas[0][0]
            radius[row][col] = np.log(rho)
            angle[row][col] = theta * 180 / np.pi +180
    return radius,angle

#获取bin
def get_bin(radius, angle, region_size):
    max_radius = np.amax(radius)
    bin = np.zeros((2,15,3))#2维 15行 3列
    for m in range(15):
        theta_low = m*24
        theta_up = (m+1)*24
        for n in range(3):
            rho_low = max_radius * n / 3
            rho_up = max_radius * (n+1) / 3
            temp = []

            for row in range(1,region_size[0]):
                for col in range(1,region_size[1]):
                    ra = radius[row][col]
                    an = angle[row][col]
                    if ra>=rho_low and ra<=rho_up and an>=theta_low and an<=theta_up:
                        bin[0][m][n],bin[1][m][n] = row,col
    return bin

#计算自相似描述子
def com_Self_Similarities(img, region_size, patch_size, bin):
    # Convert image to LAB space
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_size = lab_image.shape[:2]

    vec_size = 45  # 45 bins
    alpha = 1 / (85 ** 2)  # alpha parameter used for similarity calculation
    self_similarities = np.zeros((lab_size[0], lab_size[1], vec_size)) #w*h*45
    center_region = [region_size[0]/2, region_size[1]/2]
    center_patch = [patch_size[0]/2, patch_size[1]/2]

    for row in range(int(center_region[0] + 1), int(lab_size[0] - center_region[0])):
        for col in range(int(center_region[1] + 1), int(lab_size[1] - center_region[1])):
            patch = lab_image[int(row - center_patch[0]):int(row + center_patch[0] + 1),
                    int(col - center_patch[1]):int(col + center_patch[1] + 1), :]
            region = lab_image[int(row - center_region[0]):int(row + center_region[0] + 1),
                     int(col - center_region[1]):int(col + center_region[1] + 1), :]
            SSD_region = cal_ssd(patch, region, alpha, center_patch)
            vec = get_self_sim_vec(SSD_region, bin, vec_size)
            LSSD = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
            self_similarities[row, col, :] = LSSD

    return self_similarities

def cal_ssd(patch, region, alpha, center_patch):

    patch_size = patch.shape[:2]
    region_size = region.shape[:2]
    SSD_region = np.zeros((region_size[0],region_size[1]))

    for row in range(int(center_patch[0]),int(region_size[0]-center_patch[0]+1)):
        for col in range(int(1+center_patch[1]),int(region_size[1]-center_patch[1]+1)):
            temp = region[int(row-center_patch[0]):int(center_patch[0]+row+1)][int(col-center_patch[1]):int(center_patch[1]+col)][:] - patch[:][:][:]
            SSD_region[row][col] = np.sum(temp)
            SSD_region[row][col] = np.exp(-alpha * SSD_region[row][col])
    return SSD_region

def get_self_sim_vec(ssd_region, bin, vec_size):
    self_similarities_vec = np.zeros((1,vec_size))
    num = 0
    for m in range(15):
        for n in range(3):
            temp = bin[m][n]
            max_value = 0
            temp_shape = temp.shape[:2]
            for loc in range(temp_shape[1]):
                row = temp[0][loc-1]
                col = temp[1][loc-1]
                max_value = max(max_value,ssd_region[row-1][col-1])
            num += 1
            self_similarities_vec[num-1] = max_value
    return self_similarities_vec



#计算每张图的自相似性描述子表示
n_test = 2
region_size = [45, 37]
patch_size = [5, 5]
radius, angle = cart2polar(region_size)#转化为极坐标
bin = get_bin(radius, angle,region_size)

for m in range(0, n_img):
    src_img = thumbs[m]
    ssm = []
    imgRgb = cv2.resize(src_img, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)  # 缩小为原来的三分之一
    self_similarities = com_Self_Similarities(imgRgb, region_size, patch_size, bin)
    ssm.append(self_similarities)
#
#
# #计算显著性得分
# width = 1
# height = 1#设置子图像大小为100*100
# center_sub = [width // 2, height // 2]
# p = 1#距离度量采用L2范数
# self_similarities = [None] * n_img
# for m in range(1, n_img + 1):
#     Path = 'self_similarities0' + str(m) + '.npy'
#     temp = np.load(Path)#载入图像的自相似性描述子
#     self_similarities1 = temp
#     img_size1 = self_similarities1.shape
#     src_img = cv2.imread('Input/' + str(m) + '.jpg')
#     imgRgb = cv2.resize(src_img, None, fx=1 / 3, fy=1 / 3)
#     sig_score_img = np.zeros((img_size1[0], img_size1[1]))
#     for row1 in range(center_sub[0] + 1, img_size1[0] - center_sub[0] - 1):
#         for col1 in range(center_sub[1] + 1, img_size1[1] - center_sub[1] - 1):
#             sub1 = self_similarities1[row1 - center_sub[0]:row1 + center_sub[0] + 1,
#                    col1 - center_sub[1]:col1 + center_sub[1] + 1, :]#第一幅图像的子图像
#             max_match = np.zeros((1, n_img - 1))#记录与其他各个图像的块的最大匹配得分
#             num_img = 1
#             match_score = [None] * n_img#记录与其他各个图像的块的匹配得分
#
#             #记录与其他图像的相似性
#             for n in range(1, n_img + 1):
#                 if n != m:
#                     Path = 'self_similarities0' + str(n) + '.npy'
#                     temp = np.load(Path)
#                     self_similarities2 = temp
#                     temp1 = np.tile(sub1, (self_similarities2.shape[0], self_similarities2.shape[1], 1))
#                     temp2 = -1 * np.sum((self_similarities2 - temp1) ** 2, axis=2)
#                     max_match[0, num_img - 1] = np.max(temp2)#记录与每副图像的最大匹配得分
#                     match_score[num_img - 1] = np.reshape(temp2, (-1, 1))
#                     num_img = num_img + 1
#             temp3 = np.vstack((match_score[0], match_score[1], match_score[2], match_score[3], match_score[4]))
#             avgMatch = np.mean(temp3)#该像素点处的矩形框在其它所有图像中的平均匹配得分
#             stdMatch = np.std(temp3)#匹配得分标准差
#             sig_score_img[row1, col1] = np.sum((max_match - avgMatch) / stdMatch)
#
#     savePath = 'sig_img11' + str(m) + '.npy'
#     np.save(savePath, sig_score_img)#保存每副图像显著性得分mat
#     path = 'sig_img11' + str(m) + '.npy'
#     temp = np.load(path)
#     sig_score_img = temp.sig_score_img
#     src_img = cv2.imread(f"Input/{m}.jpg")
#     draw_result(src_img, sig_score_img / 4, [45, 37], 3)