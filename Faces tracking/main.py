import math
import numpy as np
import cv2




def get_tr(img):
    # 定义需要返回的参数
    mouse_params = {'x': None, 'width': None, 'height': None,
                    'y': None, 'temp': None}
    cv2.namedWindow('image')
    # 鼠标框选操作函数
    cv2.setMouseCallback('image', on_mouse, mouse_params)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return [mouse_params['x'], mouse_params['y'], mouse_params['width'],
            mouse_params['height']], mouse_params['temp']


def on_mouse(event, x, y, flags, param):
    global img, point1
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        # 返回框选矩形左上角点的坐标、矩形宽度、高度以及矩形包含的图像
        param['x'] = min(point1[0], point2[0])
        param['y'] = min(point1[1], point2[1])
        param['width'] = abs(point1[0] - point2[0])
        param['height'] = abs(point1[1] - point2[1])
        param['temp'] = img[param['y']:param['y'] + param['height'],
                        param['x']:param['x'] + param['width']]


def main():
    global img
    cap = cv2.VideoCapture("faces.mp4")
    # 获取视频第一帧
    ret, frame = cap.read()

    img = frame

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 创建视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vedioWrite = cv2.VideoWriter('tracted.mp4', fourcc, 25, size)

    # 框选目标并返回相应信息：rect为四个信息，temp为框选出来的图像
    rect, temp = get_tr(img)
    print(temp)
    (a, b, c) = temp.shape
    y = [a / 2, b / 2]

    # 计算目标图像的权值矩阵
    m_wei = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            z = (i - y[0]) ** 2 + (j - y[1]) ** 2
            m_wei[i, j] = 1 - z / (y[0] ** 2 + y[1] ** 2)

    # 计算目标权值直方图
    C = 1 / sum(sum(m_wei))
    hist1 = np.zeros(16 ** 3)
    for i in range(a):
        for j in range(b):
            q_b = math.floor(float(temp[i, j, 0]) / 16)
            q_g = math.floor(float(temp[i, j, 1]) / 16)
            q_r = math.floor(float(temp[i, j, 2]) / 16)
            q_temp1 = q_r * 256 + q_g * 16 + q_b
            hist1[int(q_temp1)] = hist1[int(q_temp1)] + m_wei[i, j]
    hist1 = hist1 * C

    # 接着读取视频并进行目标跟踪
    while (1):
        ret, frame = cap.read()

        if ret == True:
            Img = frame
            num = 0
            Y = [1, 1]

            # mean shift迭代
            while (np.sqrt(Y[0] ** 2 + Y[1] ** 2) > 0.5) & (num < 20):
                num = num + 1

                # 计算候选区域直方图
                temp2 = Img[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]
                hist2 = np.zeros(16 ** 3)
                q_temp2 = np.zeros((a, b))
                for i in range(a):
                    for j in range(b):
                        q_b = math.floor(float(temp2[i, j, 0]) / 16)
                        q_g = math.floor(float(temp2[i, j, 1]) / 16)
                        q_r = math.floor(float(temp2[i, j, 2]) / 16)
                        q_temp2[i, j] = q_r * 256 + q_g * 16 + q_b
                        hist2[int(q_temp2[i, j])] = hist2[int(q_temp2[i, j])] + m_wei[i, j]
                hist2 = hist2 * C

                w = np.zeros(16 ** 3)
                for i in range(16 ** 3):
                    if hist2[i] != 0:
                        w[i] = math.sqrt(hist1[i] / hist2[i])
                    else:
                        w[i] = 0

                sum_w = 0
                sum_xw = [0, 0]
                for i in range(a):
                    for j in range(b):
                        sum_w = sum_w + w[int(q_temp2[i, j])]
                        sum_xw = sum_xw + w[int(q_temp2[i, j])] * np.array([i - y[0], j - y[1]])
                Y = sum_xw / sum_w

                # 位置更新
                rect[0] = rect[0] + Y[1]
                rect[1] = rect[1] + Y[0]

            v0 = int(rect[0])
            v1 = int(rect[1])
            v2 = int(rect[2])
            v3 = int(rect[3])
            pt1 = (v0, v1)
            pt2 = (v0 + v2, v1 + v3)

            # 画矩形
            IMG = cv2.rectangle(Img, pt1, pt2, (0, 0, 255), 2)
            cv2.imshow('IMG', IMG)
            vedioWrite.write(IMG)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break


if __name__ == '__main__':
    main()