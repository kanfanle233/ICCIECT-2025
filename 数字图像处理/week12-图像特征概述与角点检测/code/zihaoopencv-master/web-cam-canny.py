
# 调用摄像头，实时边缘检测
# 同济子豪兄 2020-07-27

"""
摄像头实时 Canny 边缘检测演示

教学重点：
1. cv2.VideoCapture(0) 获取系统默认摄像头
2. cv2.Canny 对每帧画面进行实时边缘检测
3. np.dstack 将单通道边缘图堆叠为三通道，方便 imshow 正常显示彩色窗口
"""

# 导入opencv-python
import cv2

# 导入科学计算库numpy
import numpy as np

# --- 1. 初始化摄像头 ---
# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 打开cap
cap.open(0)

# --- 2. 循环读取并处理每一帧 ---
# 循环
while cap.isOpened():
    # 获取画面
    flag, frame = cap.read()  # flag: 是否成功读取，frame: 当前帧图像（BGR三通道）
    if not flag:
        break
    # 获取键盘上按下哪个键
    key_pressed = cv2.waitKey(60)  # 等待60ms，返回按键的ASCII码
    print('键盘上被按下的键是：',key_pressed)

    # frame = cv2.resize(frame, (100,100))
    # 进行canny边缘检测
    frame = cv2.Canny(frame, 100, 200)  # 低阈值100，高阈值200，输出单通道边缘图
    # 将单通道图像复制三份，摞成三通道图像
    frame = np.dstack((frame, frame, frame))  # 沿第三维度堆叠，(H,W)->(H,W,3)
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)

    # 如果按下esc键，就退出循环
    if key_pressed == 27:  # ESC键的ASCII码为27
        break

# --- 3. 释放资源 ---
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()
