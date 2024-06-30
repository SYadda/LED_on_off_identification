import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_light_up(img): 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 200, param1=10, param2=90, minRadius=75, maxRadius=125) # 0 and 1
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 200, param1=40, param2=60, minRadius=60, maxRadius=100) # 00 and 000 and 11


    # 如果检测到圆形，则在原图上绘制圆形
    if circles is not None and circles[0] is not None:
        circles = np.uint16(np.around(circles))
        # 绘制圆轮廓
        for i in circles[0, :]:
            # print(i[0], i[1]) 查看圆心坐标
            if 1300 < i[0] and i[0] < 1450 and 625 < i[1] and i[1] < 750:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 10)

                # 计算圆形区域的平均灰度值
                mask = np.zeros_like(img_gray)
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
                circle_pixels = cv2.bitwise_and(img_gray, img_gray, mask=mask)
                mean_val = np.mean(circle_pixels[circle_pixels > 0])

                # 若mean_val > 150，则认为灯亮，在圆心绘制蓝色的数字“1”，否则绘制蓝色的数字“0”
                hori_offset = 15 # 水平偏移量
                vert_offset = 5 # 垂直偏移量

                if mean_val > 150:
                    cv2.putText(img, '1', (i[0] - hori_offset, i[1] + vert_offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                else:
                    cv2.putText(img, '0', (i[0] - hori_offset, i[1] + vert_offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.show()
                return img


# 逐帧捕获
cap = cv2.VideoCapture('video.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有帧了，就退出循环
    out.write(is_light_up(frame))
cap.release()
out.release()