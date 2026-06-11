"""
快速检查 CelebA 图片文件是否存在

功能：遍历 test_df 前 50 行，检查对应图片文件是否存在，
     统计缺失数量。用于调试数据路径配置问题。
"""

import os

missing = 0
for k in range(50):
    row = test_df.iloc[k]
    img_path = os.path.join(str(IMG_DIR), row["image_id"])
    if not os.path.exists(img_path):
        missing += 1
        print("Missing:", img_path)

print("Missing in first 50:", missing)
