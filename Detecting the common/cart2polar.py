import numpy as np

# def cart2polar(region_size):
#     radius = np.zeros(region_size)
#     angle = np.zeros(region_size)
#     center = [np.ceil(region_size[0]/2), np.ceil(region_size[1]/2)]
#     for row in range(region_size[0]):
#         for col in range(region_size[1]):
#             theta,rho = np.arctan2(row-center[0], col-center[1]), np.hypot(row-center[0], col-center[1])
#             radius[row,col] = np.log(rho)
#             angle[row,col] = theta*180/np.pi + 180
#     return radius, angle

def cartesian_to_polar(img, bin):
    # 获取图像大小
    region_size = img.shape

    # 设置角度和半径的bin数
    angles = bin[0]
    radii = bin[1]

    # 计算中心点坐标
    center_x = region_size[1] // 2
    center_y = region_size[0] // 2

    # 初始化半径和角度数组
    radius = np.zeros((region_size[0], region_size[1]), dtype=np.float32)
    angle = np.zeros((region_size[0], region_size[1]), dtype=np.float32)

    # 计算每个像素的半径和角度值
    for i in range(region_size[0]):
        for j in range(region_size[1]):
            # 计算像素点到中心点的距离
            rho = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)

            # 计算像素点与x轴正方向的夹角
            theta = np.arctan2(i - center_y, j - center_x)

            # 将角度转化为0-2pi的范围
            if theta < 0:
                theta += 2 * np.pi

            # 将半径和角度存储到数组中
            radius[i][j] = np.log(rho)
            angle[i][j] = theta

    # 将角度和半径按照bin进行划分，统计每个区域内的像素数量
    radii_bins = np.linspace(radius.min(), radius.max(), radii + 1)
    angles_bins = np.linspace(0, 2 * np.pi, angles + 1)
    hist = np.zeros((radii, angles), dtype=np.float32)
    for i in range(angles):
        for j in range(radii):
            r_inner = radii_bins[j]
            r_outer = radii_bins[j + 1]
            a_inner = angles_bins[i]
            a_outer = angles_bins[i + 1]
            idx = np.where((radius >= r_inner) & (radius < r_outer) &
                           (angle >= a_inner) & (angle < a_outer))
            hist[j, i] = idx[0].shape[0]

    # 将直方图归一化
    hist /= np.sum(hist)

    # 将归一化后的直方图展开为一维数组
    vec = hist.reshape(1, angles * radii)

    return vec