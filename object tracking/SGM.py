import cv2
import numpy as np

video_capture = cv2.VideoCapture('test.avi')  # 读取视频
frame_count = int(cv2.VideoCapture('test.avi').get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
ret, frame = video_capture.read()

#创建视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vedioWrite = cv2.VideoWriter('tract_sgm.mp4', fourcc, 25, size)

alpha = 0.03
std_init = 20
var_init = std_init * std_init
lamda = 2.5 * 1.2
h = int(frame.shape[0])
w = int(frame.shape[1])
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

frame_u = np.zeros((h, w, 3))
frame_d = np.zeros((h, w, 3))
frame_std = np.zeros((h, w, 3))
frame_var = np.zeros((h, w, 3))

for i in range(h):
    for j in range(w):
        pixel_R = frame[i, j, 0]
        pixel_G = frame[i, j, 1]
        pixel_B = frame[i, j, 2]

        pixel_uR = pixel_R
        pixel_uG = pixel_G
        pixel_uB = pixel_B

        pixel_dR, pixel_dG, pixel_dB = 0, 0, 0

        pixel_stdR, pixel_stdG, pixel_stdB = std_init, std_init, std_init

        pixel_varR, pixel_varG, pixel_varB = var_init, var_init, var_init

        frame_u[i, j, 0] = pixel_uR
        frame_u[i, j, 1] = pixel_uG
        frame_u[i, j, 2] = pixel_uB
        frame_d[i, j, 0] = pixel_dR
        frame_d[i, j, 1] = pixel_dG
        frame_d[i, j, 2] = pixel_dB
        frame_std[i, j, 0] = pixel_stdR
        frame_std[i, j, 1] = pixel_stdG
        frame_std[i, j, 2] = pixel_stdB
        frame_var[i, j, 0] = pixel_varR
        frame_var[i, j, 1] = pixel_varG
        frame_var[i, j, 2] = pixel_varB

while (True):
    for k in range(1, frame_count, 5):  # 读取帧数据
        ret, frame = video_capture.read()
        for i in range(h):
            for j in range(w):
                pixel_R = frame[i, j, 0]
                pixel_G = frame[i, j, 1]
                pixel_B = frame[i, j, 2]

                pixel_uR = frame_u[i, j, 0]
                pixel_uG = frame_u[i, j, 1]
                pixel_uB = frame_u[i, j, 2]

                pixel_dR = frame_d[i, j, 0]
                pixel_dG = frame_d[i, j, 1]
                pixel_dB = frame_d[i, j, 2]

                pixel_stdR = frame_std[i, j, 0]
                pixel_stdG = frame_std[i, j, 1]
                pixel_stdB = frame_std[i, j, 2]

                pixel_varR = frame_var[i, j, 0]
                pixel_varG = frame_var[i, j, 1]
                pixel_varB = frame_var[i, j, 2]

                if ((abs(pixel_R - pixel_uR) < lamda * pixel_stdR) and
                        (abs(pixel_G - pixel_uG) < lamda * pixel_stdG) and
                        (abs(pixel_B - pixel_uB) < lamda * pixel_stdB)):

                    pixel_uR = (1 - alpha) * pixel_uR + alpha * pixel_R
                    pixel_uG = (1 - alpha) * pixel_uG + alpha * pixel_G
                    pixel_uB = (1 - alpha) * pixel_uB + alpha * pixel_B

                    pixel_varR = (1 - alpha) * pixel_varR + alpha * (pixel_R - pixel_uR) * (pixel_R - pixel_uR)
                    pixel_varG = (1 - alpha) * pixel_varG + alpha * (pixel_G - pixel_uG) * (pixel_G - pixel_uG)
                    pixel_varB = (1 - alpha) * pixel_varB + alpha * (pixel_B - pixel_uB) * (pixel_B - pixel_uB)

                    pixel_stdR = np.sqrt(np.double(pixel_varR))
                    pixel_stdG = np.sqrt(np.double(pixel_varG))
                    pixel_stdB = np.sqrt(np.double(pixel_varB))

                    frame_u[i, j, 0] = pixel_uR
                    frame_u[i, j, 1] = pixel_uG
                    frame_u[i, j, 2] = pixel_uB
                    frame_d[i, j, 0] = pixel_dR
                    frame_d[i, j, 1] = pixel_dG
                    frame_d[i, j, 2] = pixel_dB
                    frame_std[i, j, 0] = pixel_stdR
                    frame_std[i, j, 1] = pixel_stdG
                    frame_std[i, j, 2] = pixel_stdB
                else:
                    pixel_dR = pixel_R - pixel_uR
                    pixel_dG = pixel_G - pixel_uG
                    pixel_dB = pixel_B - pixel_uB
                    frame_d[i, j, 0] = pixel_dR
                    frame_d[i, j, 1] = pixel_dG
                    frame_d[i, j, 2] = pixel_dB

    cv2.imshow('frame_d', frame_d)
    cv2.imshow('frame_u', frame_u)
    vedioWrite.write(frame_d)#将处理后的帧写入视频
    cv2.waitKey()

video_capture.release()
cv2.destroyAllWindows()
