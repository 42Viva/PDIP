import cv2
import numpy as np

def  harris_corner(image, blockSize, ksize, k, threshold):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    R = cv2.cornerHarris(gray_image,blockSize,ksize,k)

    corner_image=np.array(image)
    # corner_image = np.array(gray_image)
    corner_image[R > threshold * R.max()] = [0, 255, 0]

    return corner_image

if __name__ == '__main__':
    image = cv2.imread(r'r2.jpg')
    blocksize = 2
    ksize = 3
    k = 0.04
    threshold = 0.01
    corner_image = harris_corner(image,blocksize,ksize,k,threshold)
    cv2.imshow('Harris Corner Image',corner_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()