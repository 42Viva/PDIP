import cv2,os
import numpy as np

img_path = r'E:\Projects\CVpractice\Exp2\imgs'
imgs = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
n_img = len(imgs)
thumbs = []
for i in range(n_img):
    thumbs.append(cv2.imread(os.path.join(img_path, imgs[i])))



# cartToPolar函数实现
def cartToPolar(img, patch_size, bin):
    h, w = img.shape[:2]  # 获取图像的高度和宽度
    center = (int(w/2), int(h/2))  # 获取图像中心点坐标
    polar_img = np.zeros((bin[0]*bin[1], patch_size*patch_size))  # 极坐标图像
    count = 0  # 极坐标图像中的像素点计数器
    # 循环遍历每个bin
    for i in range(bin[0]):
        # 计算当前bin的角度范围
        start_angle = i*360.0/bin[0]
        end_angle = (i+1)*360.0/bin[0]
        # 循环遍历每个bin中的radial interval
        for j in range(bin[1]):
            # 计算当前radial interval的长度范围
            start_length = j*patch_size/bin[1]
            end_length = (j+1)*patch_size/bin[1]
            # 极坐标变换
            polar_patch = cv2.linearPolar(img, center, end_length, cv2.WARP_FILL_OUTLIERS+cv2.INTER_LINEAR+cv2.WARP_POLAR_LINEAR)
            polar_patch = polar_patch[int(center[1]-end_length):int(center[1]-start_length), int(center[0]+start_angle):int(center[0]+end_angle)]
            # 将极坐标图像中的像素点复制到一维数组中
            polar_img[count] = polar_patch.flatten()
            count += 1
    return polar_img


# 计算自相似性描述子
def get_self_similarities(img, patch_size, bin):
    # 获取图像的高度和宽度
    height, width = img.shape[:2]

    # 计算图像中所有可能的路径数
    path_num = (height - patch_size + 1) * (width - patch_size + 1)

    # 初始化自相似性描述子向量
    ssd_vec = np.zeros((path_num, bin), dtype=np.float32)

    # 对于每个可能的路径，计算它的自相似性描述子
    for i in range(height - patch_size + 1):
        for j in range(width - patch_size + 1):
            # 提取路径中的补丁
            patch = img[i:i+patch_size, j:j+patch_size]

            # 计算补丁的自相似性矩阵
            ssd_region = cv2.matchTemplate(patch, patch, cv2.TM_CCORR_NORMED)

            # 将自相似性矩阵转换为自相似性描述子向量
            ssd_vec[i*(width-patch_size+1)+j] = get_self_sim_vec(ssd_region, bin, bin)

    return ssd_vec

# 计算自相似性描述子向量
def get_self_sim_vec(ssd_region, bin, vec_size):
    # 将自相似性矩阵分成几个bin
    bins = np.linspace(-1, 1, bin+1)

    # 计算每个bin的计数
    hist, _ = np.histogram(ssd_region, bins=bins)

    # 计算自相似性描述子向量
    ssd_vec = np.zeros((vec_size,), dtype=np.float32)
    for i in range(vec_size):
        ssd_vec[i] = np.sum(hist[i*bin//vec_size:(i+1)*bin//vec_size])

    # 归一化自相似性描述子向量
    ssd_vec /= np.sum(ssd_vec)

    return ssd_vec

# 检测多张图片中的共同部分
def detect_common(img_list, patch_size, bin, threshold):
    # 将每张图像转换为极坐标
    polar_img_list = []
    for img in img_list:
        polar_img = cartToPolar(img, patch_size, bin)
        polar_img_list.append(polar_img)

    # 对每张图像计算自相似性描述子
    ssd_list = []
    for polar_img in polar_img_list:
        ssd = get_self_similarities(polar_img, patch_size, bin)
        ssd_list.append(ssd)

    # 计算每张图像和其他图像的相似度
    sim_list = []
    num_imgs = len(img_list)
    for i in range(num_imgs):
        sim = np.zeros(num_imgs)
        for j in range(num_imgs):
            if i == j:
                sim[j] = 1.0
            else:
                sim[j] = get_similarity(ssd_list[i], ssd_list[j])
        sim_list.append(sim)

    # 找出相似度超过阈值的图像
    common_imgs = []
    for i in range(num_imgs):
        if sum(sim_list[i] >= threshold) > 1:
            common_imgs.append(i)

    # 找出每张相似图像中的共同部分
    common_patches = []
    for img_idx in common_imgs:
        for i in range(len(ssd_list[img_idx])):
            if np.all(sim_list[img_idx] >= threshold):
                common_patches.append((img_idx, i))

    return common_patches


def get_similarity(ssd1, ssd2):
    """
    Compute the similarity score between two self-similarity descriptors.

    Args:
        ssd1 (ndarray): The self-similarity descriptor for the first image.
        ssd2 (ndarray): The self-similarity descriptor for the second image.

    Returns:
        float: The similarity score between the two descriptors.
    """
    # Compute the cosine similarity between the two descriptors
    norm1 = np.linalg.norm(ssd1)
    norm2 = np.linalg.norm(ssd2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cosine_sim = np.dot(ssd1, ssd2) / (norm1 * norm2)

    # Clip the cosine similarity to the range [0, 1] to avoid numerical errors
    return np.clip(cosine_sim, 0, 1)


print(detect_common(thumbs,5 ,(15,3) ,0.2))