import cv2
import numpy as np

video = cv2.VideoCapture(r'E:\Projects\CVpractice\Exp3\test.avi')
# 判断视频是否打开
if (video.isOpened()):
    print('视频读取成功')
else:
    print('视频读取失败')

# 测试用,查看视频size
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:' + repr(size))

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None

#创建视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vedioWrite = cv2.VideoWriter('tract.mp4', fourcc, 25, size)



while True:
    # 读取视频流
    grabbed, frame_CV = video.read()
    if grabbed is False:
        print('Mission Finsh')
        break
    gray_CV = cv2.cvtColor(frame_CV, cv2.COLOR_BGR2GRAY)# 对帧进行预处理，先转灰度图，再进行高斯滤波。
    gray_CV = cv2.GaussianBlur(gray_CV, (21, 21), 0)
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_CV
        continue
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_CV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=1)  # 形态学膨胀

    # 显示矩形框
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) < 1600:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(frame_CV, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('contours', frame_CV)
    cv2.imshow('dis', diff)

    vedioWrite.write(frame_CV)#将处理后的帧写入视频

    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()