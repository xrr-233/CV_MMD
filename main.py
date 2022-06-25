import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

if(__name__=="__main__"):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=True,
                        model_complexity=2,
                        smooth_landmarks=True,
                        enable_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)


    for pic in os.listdir(f'{os.getcwd()}/images'):
        if not(pic.endswith('jpg') or pic.endswith('png')):
            continue

        print(pic)

        img = cv_imread(f'{os.getcwd()}/images/{pic}')
        # look_img(img)

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(img_RGB)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        look_img(img)
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # 1.二次元效果不佳
    # 2.三次元只适合单人