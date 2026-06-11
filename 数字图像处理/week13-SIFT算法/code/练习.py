# 练习.py   放在 week13-SIFT算法/code/ 下

"""
基于 LBPH 的 ORL 人脸数据集识别实验

教学重点：
1. ORL 数据集包含 40 人各 10 张灰度人脸图，每张 112x92 像素
2. cv2.face.LBPHFaceRecognizer_create() 创建 LBPH（局部二值模式直方图）人脸识别器
3. 每人最后一张做测试，其余做训练，计算整体识别准确率
4. 置信度（confidence）为欧氏距离，值越小表示越相似
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# --- 1. 数据加载函数 ---
def load_orl_dataset(dataset_path):
    """
    加载 ORL 人脸数据集，返回训练图像/标签与测试图像/标签

    参数:
        dataset_path: ORL 数据集根目录路径，每个子目录代表一个人
    返回:
        faces:       训练图像数组 (N, 100, 100)，uint8 灰度图
        labels:      训练标签数组 (N,)，每人一个整数编号
        test_faces:  测试图像列表，每人最后一张
        test_labels: 测试标签列表
    """

    faces = []       # 训练图像
    labels = []      # 训练标签
    test_faces = []  # 测试图像
    test_labels = [] # 测试标签

    # 只用真正的图片文件，过滤掉 ._ 开头和乱七八糟的 pgm
    valid_ext = [".bmp", ".jpg", ".jpeg", ".png"]  # 你现在主要是 bmp

    for label, person in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        # 过滤出有效图片：不以 ._ 开头，后缀在 valid_ext 里
        img_files = [
            f for f in os.listdir(person_path)
            if (not f.startswith("._")) and os.path.splitext(f)[1].lower() in valid_ext
        ]
        img_files.sort()

        if len(img_files) == 0:
            print("该文件夹没有有效图片：", person_path)
            continue

        # 最后一张作为测试
        test_name = img_files[-1]
        test_path = os.path.join(person_path, test_name)
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (100, 100))
        test_faces.append(test_img)
        test_labels.append(label)

        # 其余作为训练
        for img_name in img_files[:-1]:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("训练图读取失败：", img_path)
                continue
            img = cv2.resize(img, (100, 100))
            faces.append(img)
            labels.append(label)

    faces = np.array(faces, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    return faces, labels, test_faces, test_labels


# --- 2. 主函数 ---
def main():
    # 1. 路径（你现在的真实位置）
    dataset_path = "img/ORLdataset"

    # 2. 加载数据
    faces, labels, test_faces, test_labels = load_orl_dataset(dataset_path)
    print("训练集图像数量：", len(faces))
    print("测试集图像数量：", len(test_faces))

    # 3. 创建并训练 LBPH 识别器
    # LBPH（Local Binary Patterns Histograms）对光照变化有较好鲁棒性
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    print("模型训练完成！")

    # 4. 在测试集上计算准确率
    # predict 返回 (预测标签, 置信度)，置信度为欧氏距离，越小越相似
    correct = 0
    for i, img in enumerate(test_faces):
        pred_label, conf = recognizer.predict(img)
        if pred_label == test_labels[i]:
            correct += 1

    accuracy = correct / len(test_faces) * 100
    print(f"预测正确：{correct}/{len(test_faces)}，识别率：{accuracy:.2f}%")

    # 5. 随机挑一张测试图显示结果
    idx = 0  # 想看哪个就改成几，0~len(test_faces)-1
    test_img = test_faces[idx]
    true_label = test_labels[idx]

    pred_label, conf = recognizer.predict(test_img)
    true_text = f"s{true_label + 1}"
    pred_text = f"s{pred_label + 1}" if pred_label != -1 else "Unknown"

    display = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, f"Pred: {pred_text}", (5, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    plt.imshow(display_rgb)
    plt.title(f"True: {true_text}   Pred: {pred_text}\nConf: {conf:.2f}")
    plt.axis("off")
    plt.show()

    print("单张测试：")
    print("  真实类别：", true_text)
    print("  预测类别：", pred_text)
    print("  置信度（距离，越小越像）：", conf)


if __name__ == "__main__":
    main()
