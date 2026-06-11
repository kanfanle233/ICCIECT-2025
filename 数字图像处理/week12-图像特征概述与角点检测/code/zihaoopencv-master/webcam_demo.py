
# 调用摄像头，实时展示画面
# 同济子豪兄 2020-07-27

"""
摄像头实时画面展示基础演示

教学重点：
1. cv2.VideoCapture 打开摄像头获取视频流
2. cap.read() 循环读取每一帧画面
3. cv2.waitKey 控制显示帧率并监听键盘事件
"""

# 导入opencv-python
import cv2

# --- 1. 初始化摄像头 ---
# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0=默认摄像头，CAP_DSHOW=DirectShow后端
# 打开cap
cap.open(0)

# --- 2. 循环读取并显示画面 ---
# 循环
while cap.isOpened():
    # 获取画面
    flag, frame = cap.read()  # flag=是否成功，frame=BGR图像

    cv2.imshow('my_window',frame)

    # 获取键盘上按下哪个键
    key_pressed = cv2.waitKey(60)  # 等待60ms，控制帧率约16fps
    print('键盘上被按下的键是：',key_pressed)
    # 如果按下esc键，就退出循环
    if key_pressed == 27:  # ESC键的ASCII码为27
        break

# --- 3. 释放资源 ---
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()
