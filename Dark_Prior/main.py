import cv2
import numpy as np
import sys
import copy
from scipy.sparse import diags


# 计算图像的暗通道
def dark_channel(img, w):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((2 * w + 1), (2 * w + 1)))
    dc_img = cv2.erode(min_img, kernel)

    return dc_img

# def dark_channel(img, w):
#     # 将图像转换为浮点数
#     image = img.astype(np.float32) / 255.0
#
#     # 使用矩形滤波器来求解暗通道
#     kernel = np.ones(((2*w+1), (2*w+1)), np.float32) / ((2*w+1) * (2*w+1))
#     dark_channel = cv2.dilate(image, kernel)
#     dark_channel = np.min(dark_channel, axis=2)
#
#     return dark_channel

# 计算大气光A
def estimating_atmospheric_light(image, dark_channel):
    i = dark_channel.shape
    num_sum = dark_channel.shape[0] * dark_channel.shape[1] #总像素数
    num = int(num_sum * 0.001)#应选出的像素数量（前99.9%）
    print("image has " + str(num_sum) + " pixels. Top 0.1% has " + str(num) + " points")

    pixels = []  # 储存全部的像素值
    for i in range(dark_channel.shape[0]):
        for j in range(dark_channel.shape[1]):
            pixels.append(dark_channel[i][j])

    pixels_sorted_id = sorted(range(len(pixels)), key=lambda x: pixels[x])  # 像素值排序，储存升序后的id

    value_threshold = pixels[pixels_sorted_id[len(pixels)-num]]
    print("dark channel value >= " + str(value_threshold) + " belong 0.1%")

    max_value = 0
    row = 0
    col = 0
    for i in range(dark_channel.shape[0]):
        for j in range(dark_channel.shape[1]):
            if dark_channel[i][j] > value_threshold:
                value = int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2])#在原图中确定最亮点
                if value > max_value:
                    max_value = value
                    row = i
                    col = j

    A = image[row][col]
    print("A at: [" + str(row) + ", " + str(col) + "], value: " + str(A))
    return A



# def softmatting(Img, tmap, win_size, epsilon, lam=0.001):
#
#     if epsilon is None:
#         epsilon = 0.0001
#     if win_size is None:
#         win_size = 1
#     if lam is None:
#         lam = 0.001
#
#     h, w = Img.shape
#     img_size = w * h
#
#     win_b = np.zeros((img_size, 1))
#
#     for i in range(h):
#         for j in range(w):
#             if (i - 8) % 15 < 1 and (j - 8) % 15 < 1:
#                 win_b[i * w + j] = tmap[i,j]
#
#     # get Laplacian matrix L
#     L = cv2.Laplacian(Img,cv2.CV_64F, epsilon, win_size)
#
#     # solve for refined transmission map
#     D = diags(win_b.ravel(), 0, (img_size, img_size))
#     x = np.linalg.solve(L + lam * D, lam * win_b.ravel() ** 2)
#     tmap_ref = np.clip(x.reshape(h, w), 0, 1)
#
#     return tmap_ref

def dehaze(image):
    dark_prior = dark_channel(image,7)  # 原图像的暗通道
    cv2.imshow("dark channel of origin", dark_prior)

    A = estimating_atmospheric_light(image, dark_prior)  # 确定大气光
    t = image / A
    t_min = dark_channel(t, 7)

    # 导向滤波优化image/A的暗通道
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    t_min = softmatting(img_gray, t_min, 1, 0.0001)  # 传播率图
    cv2.imshow("transmission", t_min)

    # 计算透射率
    t_x = [[0 for i in range(t_min.shape[1])] for i in range(t_min.shape[0])]
    for i in range(t_min.shape[0]):
        for j in range(t_min.shape[1]):
            t_x[i][j] = 1 - 0.95 * t_min[i][j]

    # 计算去雾图
    image_noHaze = copy.copy(image)
    for i in range(image_noHaze.shape[0]):
        for j in range(image_noHaze.shape[1]):
            image_noHaze[i][j][0] = max(min((int(image[i][j][0]) - int(A[0])) / max(0.1, t_x[i][j]) + int(A[0]), 255), 0)
            image_noHaze[i][j][1] = max(min((int(image[i][j][1]) - int(A[1])) / max(0.1, t_x[i][j]) + int(A[1]), 255), 0)
            image_noHaze[i][j][2] = max(min((int(image[i][j][2]) - int(A[2])) / max(0.1, t_x[i][j]) + int(A[2]), 255), 0)

    image_noHaze = np.uint8(image_noHaze)

    return image_noHaze


if __name__ == '__main__':
    image_origin = cv2.imread('foggy2.jpg')
    print("origin image shape: " + str(image_origin.shape))
    result = dehaze(image_origin)

    cv2.imshow("origin", image_origin)
    cv2.imshow("result", result)
    cv2.imwrite('defog.jpg', result)

    while True:
        k = cv2.waitKey(1)
        if k == 27:  # 按esc退出
            break
    sys.exit(0)