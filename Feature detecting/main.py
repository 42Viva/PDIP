import cv2
import numpy as np

def  harris_corner(image, blockSize, ksize, k, threshold):

    # 利用水平、竖直差分算子对图像每个像素进行被以求得Ix、Iy，进而求得m中四个元素的值:
    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)

    # 计算梯度乘积
    IxIx = I_x * I_x
    IxIy = I_x * I_y
    IyIy = I_y * I_y

    # 对m的四个元素进行高斯平滑滤波，得到新m
    IxIx = cv2.GaussianBlur(IxIx, (3, 3), 0)
    IyIy = cv2.GaussianBlur(IyIy, (3, 3), 0)
    IxIy = cv2.GaussianBlur(IxIy, (3, 3), 0)

    # 利用m计算每个像素的角点量cim（R
    det = IxIx * IyIy - IxIy * IxIy
    trace = IxIx + IyIy
    harris_response = det - k * trace * trace

    # 筛选出cim大于阈值和是局部极大值条件的点
    corners = harris_response > threshold * harris_response.max()

    corner_positions = np.argwhere(corners)
    return corner_positions

if __name__ == '__main__':
    image = cv2.imread(r'r2.jpg')

    blocksize = 2
    ksize = 3
    k = 0.04
    threshold = 0.01

    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    corner_positions = harris_corner(gray_image,blocksize,ksize,k,threshold)

    for y, x in corner_positions:
        cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
    cv2.imshow('Harris Corner Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()