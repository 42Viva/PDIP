import cv2
import numpy as np
import time

def com_Self_Similarities(src_image, region_size, patch_size, bin):
    t1 = time.time()  # Start timer

    # Convert image to LAB space
    lab_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB)

    # Calculate self-similarities descriptor for each pixel
    lab_size = lab_image.shape[:2]
    vec_size = 45  # 45 bins
    alpha = 1 / (85 ** 2)  # alpha parameter used for similarity calculation
    self_similarities = np.zeros((lab_size[0], lab_size[1], vec_size))
    center_region = [region_size[0] // 2, region_size[1] // 2]
    center_patch = [patch_size[0] // 2, patch_size[1] // 2]


    for row in range(center_region[0] + 1, lab_size[0] - center_region[0]):
        for col in range(center_region[1] + 1, lab_size[1] - center_region[1]):
            patch = lab_image[row - center_patch[0]:row + center_patch[0] + 1,
                              col - center_patch[1]:col + center_patch[1] + 1, :]
            region = lab_image[row - center_region[0]:row + center_region[0] + 1,
                               col - center_region[1]:col + center_region[1] + 1, :]
            SSD_region = cal_ssd(patch, region, alpha, center_patch)
            vec = get_self_sim_vec(SSD_region, bin, vec_size)
            LSSD = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
            self_similarities[row, col, :] = LSSD

    t2 = time.time()
    print('Elapsed time:', t2 - t1)

    return self_similarities

def cal_ssd(patch, region, alpha, center_patch):
    patch_size = patch.shape[:2]
    SSD_region = np.zeros((patch_size[0] - center_patch[0] * 2, patch_size[1] - center_patch[1] * 2))
    for i in range(center_patch[0], patch_size[0] - center_patch[0]):
        for j in range(center_patch[1], patch_size[1] - center_patch[1]):
            patch_ij = patch[i - center_patch[0]:i + center_patch[0] + 1, j - center_patch[1]:j + center_patch[1] + 1, :]
            SSD_region[i - center_patch[0], j - center_patch[1]] = np.sum(np.square(patch_ij - region)) / (patch_size[0] * patch_size[1] * alpha)
    return SSD_region

def get_self_sim_vec(SSD_region, bin, vec_size):
    min_val = np.min(SSD_region)
    max_val = np.max(SSD_region)
    bin_size = (max_val - min_val) / bin
    vec = np.zeros(vec_size)
    for i in range(vec_size):
        bin_val = min_val + i * bin_size
        vec[i] = np.sum((SSD_region >= bin_val) & (SSD_region < bin_val + bin_size))
    return vec
