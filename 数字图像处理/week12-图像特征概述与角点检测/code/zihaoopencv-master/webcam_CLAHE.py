"""
摄像头实时 CLAHE（限制对比度自适应直方图均衡）演示

教学重点：
1. CLAHE 在 HSV 色彩空间的 V（亮度）通道上进行自适应直方图均衡
2. 仅对亮度通道做均衡化，保留 H（色相）和 S（饱和度）不变，避免颜色失真
3. cv2.createCLAHE 参数：clipLimit 控制对比度放大倍数，tileGridSize 控制局部区域大小
"""

# 导入opencv-python
import cv2

# --- 1. 创建 CLAHE 对象并定义帧处理函数 ---
# 处理帧函数
clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(6,6))  # clipLimit=0表示不裁剪对比度，tileGridSize为局部网格大小
def process_frame(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v_clahe = clahe.apply(v)
    clahe_hsv_img = cv2.merge((h, s, v_clahe))
    frame = cv2.cvtColor(clahe_hsv_img, cv2.COLOR_HSV2BGR)
    return frame
  
  # --- 2. 初始化摄像头并进入主循环 ---
  # 调用摄像头，实时CLAHE直方图变换
# 同济子豪兄 2021-4-5

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    flag, frame = cap.read()
    if not flag:
        break
    
    ## 处理帧函数
    frame = process_frame(frame)
    
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)

    if cv2.waitKey(1) in [113,27]: # 按键盘上的q或esc退出
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()
