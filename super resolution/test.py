import cv2
inputs = []
targets = []
for i in range(1,6):
    img1 = cv2.imread('./set5/set5/img_00{}.png'.format(i))
    # img1 = cv2.imread(f'./set5/set5/img_00{i}.png')
    img2 = cv2.imread('./set5/set5/test{}.png'.format(i))

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    inputs.append(img1)
    targets.append(img2)

    psnr = cv2.PSNR(img1, img2)
    print('第{}对图片间的psnr为：'.format(i),psnr)