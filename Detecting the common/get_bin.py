import numpy as np

def get_bins(radius, angle, region_size):
    max_radius = np.max(radius)  # 最大半径
    bin = [[] for _ in range(15)]  # 15个bin
    for m in range(15):
        theta_low = m * 24
        theta_up = (m + 1) * 24
        for n in range(3):
            rho_low = max_radius * n / 3
            rho_up = max_radius * (n + 1) / 3
            # 循环整个region，找到属于同一个bin的图像位置，保存到list中
            temp = []
            for row in range(region_size[0]):
                for col in range(region_size[1]):
                    if (radius[row, col] >= rho_low) and (radius[row, col] <= rho_up) and \
                       (angle[row, col] >= theta_low) and (angle[row, col] <= theta_up):
                        temp.append([row, col])
            bin[m].append(temp)
    return bin