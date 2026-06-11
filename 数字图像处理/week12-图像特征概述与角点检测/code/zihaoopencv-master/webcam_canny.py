"""
摄像头实时 Canny 边缘检测（简洁版）

教学重点：
1. 实时读取摄像头画面并应用 Canny 边缘检测
2. 将单通道边缘检测结果堆叠为三通道以正常显示
"""

import cv2
import numpy as np

# --- 1. 初始化摄像头 ---
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0=默认摄像头，CAP_DSHOW=DirectShow后端（Windows）
cap.open(0)

# --- 2. 循环读取并处理每一帧 ---
while cap.isOpened():
    flag, frame = cap.read()  # flag=是否成功读取，frame=当前帧BGR图像
    if not flag:
        break
    key_pressed = cv2.waitKey(60)  # 等待60ms，返回按键ASCII码

    # frame = cv2.resize(frame, (100,100))
    frame = cv2.Canny(frame, 100, 200)  # Canny边缘检测，低阈值100，高阈值200
    frame = np.dstack((frame, frame, frame))  # 单通道堆叠为三通道，便于正常显示
    cv2.imshow('frame2',frame)

    # Break if escape key pressed
    if key_pressed == 27:  # ESC键退出
        break

# --- 3. 释放资源 ---
cap.release()
cv2.destroyAllWindows()
