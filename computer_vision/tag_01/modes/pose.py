"""姿态模式：检测人体关键点，并根据关键点几何关系做简单动作判断。"""

import cv2
import os
import sys

try:
    from .. import config
    from ..utils.helpers import draw_skeleton, calculate_angle
except ImportError:
    # Fallback for running this file directly: python pose.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config
    from utils.helpers import draw_skeleton, calculate_angle


def detect_action(keypoints):
    """根据关键点坐标判断简单动作。

    这里没有上复杂分类模型，而是直接比较“点和点之间的相对位置”。
    这样写的好处是规则透明，初学者可以直接看到“为什么被判成举手或下蹲”。
    """
    kp = config.ACTION_KEYPOINT_INDICES
    actions = []

    l_shoulder = (float(keypoints[kp["left_shoulder"]][0]),
                  float(keypoints[kp["left_shoulder"]][1]))
    r_shoulder = (float(keypoints[kp["right_shoulder"]][0]),
                  float(keypoints[kp["right_shoulder"]][1]))
    l_wrist = (float(keypoints[kp["left_wrist"]][0]),
               float(keypoints[kp["left_wrist"]][1]))
    r_wrist = (float(keypoints[kp["right_wrist"]][0]),
               float(keypoints[kp["right_wrist"]][1]))
    l_hip = (float(keypoints[kp["left_hip"]][0]),
             float(keypoints[kp["left_hip"]][1]))
    r_hip = (float(keypoints[kp["right_hip"]][0]),
             float(keypoints[kp["right_hip"]][1]))
    l_knee = (float(keypoints[kp["left_knee"]][0]),
              float(keypoints[kp["left_knee"]][1]))
    r_knee = (float(keypoints[kp["right_knee"]][0]),
              float(keypoints[kp["right_knee"]][1]))
    l_ankle = (float(keypoints[kp["left_ankle"]][0]),
               float(keypoints[kp["left_ankle"]][1]))
    r_ankle = (float(keypoints[kp["right_ankle"]][0]),
               float(keypoints[kp["right_ankle"]][1]))
    l_elbow = (float(keypoints[kp["left_elbow"]][0]),
               float(keypoints[kp["left_elbow"]][1]))
    r_elbow = (float(keypoints[kp["right_elbow"]][0]),
               float(keypoints[kp["right_elbow"]][1]))

    l_wrist_shoulder_dy = abs(l_wrist[1] - l_shoulder[1])
    r_wrist_shoulder_dy = abs(r_wrist[1] - r_shoulder[1])

    if (l_wrist_shoulder_dy < config.T_POSE_THRESH and
            r_wrist_shoulder_dy < config.T_POSE_THRESH):
        actions.append("T-POSE")

    l_knee_hip_dx = abs(l_knee[0] - l_hip[0])
    r_knee_hip_dx = abs(r_knee[0] - r_hip[0])
    l_hip_ankle_dy = abs(l_hip[1] - l_ankle[1])
    r_hip_ankle_dy = abs(r_hip[1] - r_ankle[1])

    if (l_knee_hip_dx > config.SQUAT_KNEE_HIP_HORIZONTAL_THRESH and
            l_hip_ankle_dy < config.SQUAT_HIP_ANKLE_VERTICAL_THRESH):
        actions.append("SQUAT")
    elif (r_knee_hip_dx > config.SQUAT_KNEE_HIP_HORIZONTAL_THRESH and
          r_hip_ankle_dy < config.SQUAT_HIP_ANKLE_VERTICAL_THRESH):
        actions.append("SQUAT")

    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if l_wrist[1] < l_shoulder[1] - config.RAISE_HAND_THRESH:
        actions.append("L-RAISE")
    if r_wrist[1] < r_shoulder[1] - config.RAISE_HAND_THRESH:
        actions.append("R-RAISE")

    return actions


def check_keypoints_valid(keypoints):
    """检查关键点中置信度达标的数量是否满足绘制骨架的最低要求。"""
    count = sum(1 for kp in keypoints if float(kp[2]) >= config.POSE_CONF_THRESH)
    return count >= config.MIN_KEYPOINTS_FOR_SKELETON


def process_pose(frame, model, device="cpu"):
    """对单帧图像执行姿态估计，绘制骨架并识别简单动作。"""
    results = model(
        frame,
        conf=config.POSE_CONF_THRESH,
        iou=config.POSE_IOU_THRESH,
        device=device,
        verbose=False,
    )
    result = results[0]

    if result.keypoints is not None and result.boxes is not None:
        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, kps in enumerate(keypoints_data):
            if not check_keypoints_valid(kps):
                continue

            points = draw_skeleton(frame, kps)
            actions = detect_action(kps)

            if actions:
                x1, y1 = int(boxes[i][0]), int(boxes[i][1])
                action_text = " | ".join(actions)
                cv2.putText(frame, action_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame
