# ================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ================================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

yolo = Load_Yolo_model()
tic = time.perf_counter()
with open('fastly_benchmark_yolov3.txt', 'w+') as file:
    for i in range(20):
        image_path = "./IMAGES/images/maksssksksss{}.png".format(i + 11)
        detect_image(yolo, image_path, "./IMAGES/maksssksksss11out.png", input_size=YOLO_INPUT_SIZE, show=True,
                     CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
        toc = time.perf_counter()
        file.write(f"{toc - tic:0.4f}\n")
    print(f"YOLOv3 finished in {toc - tic:0.4f} seconds")
