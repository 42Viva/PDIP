import numpy as np
import cv2,sys,os


if __name__ == '__main__':

    # Load images
    img_path = r'E:\Projects\CVpractice\Exp2\imgs'
    imgs = [i for i in os.listdir(img_path) if i.endswith('.jpg')]
    NI = len(imgs)
    thumbs = []

    for i in range(NI):
        thumbs.append(cv2.imread(os.path.join(img_path, imgs[i])))

    #read center of shape for each image
    cntr = np.load(os.path.join(img_path, 'peace_cntr.npy'))


    cv2.waitKey(0)
    cv2.destroyAllWindows()