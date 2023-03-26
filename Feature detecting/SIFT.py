import cv2

if __name__ == '__main__':
    image = cv2.imread(r'r2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp = sift.detect(gray_image, None)
    img = cv2.drawKeypoints(gray_image, kp, image)

    # cv2.imwrite('Sift_Image', img)
    cv2.imshow('Sift Corner Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()