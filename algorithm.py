import tensorflow as tf

from oldcare.facial import FaceUtil
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from oldcare.utils import fileassistant
from keras.preprocessing.image import img_to_array
import cv2
import time
import numpy as np
import os
import imutils
import subprocess
from oldcare.conv import MiniVGGNet

# 全局常量
FACIAL_EXPRESSION_TARGET_WIDTH = 28
FACIAL_EXPRESSION_TARGET_HEIGHT = 28

FALL_DETECTION_TARGET_WIDTH = 64
FALL_DETECTION_TARGET_HEIGHT = 64

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# people_info_path = 'info/people_info.csv'
# facial_expression_info_path = 'info/facial_expression_info.csv'
# facial_recognition_model_path = 'models/face_recognition_hog.pickle'
# facial_expression_model_path = 'models/face_expression.hdf5'
# fall_detection_model_path = 'models/fall_detection.hdf5'
# output_stranger_path = 'supervision/strangers'
# output_smile_path = 'supervision/smile'
# python_path = '/home/reed/anaconda3/envs/tensorflow/bin/python'

ANGLE = 20

class Algorithm(object):
    def __init__(self,
                 people_info_path,
                 facial_expression_info_path,
                 facial_expression_model_path,
                 facial_recognition_model_path,
                 fall_detection_model_path):

        # infos
        self.id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)
        self.facial_expression_id_to_name = fileassistant.get_facial_expression_info(facial_expression_info_path)

        # models
        self.facial_recognition_model = FaceUtil(facial_recognition_model_path)
        self.facial_expression_model = MiniVGGNet.build(width=FACIAL_EXPRESSION_TARGET_WIDTH,
                                                   height=FACIAL_EXPRESSION_TARGET_HEIGHT,
                                                   depth=1, classes=2)
        self.facial_expression_model.load_weights(facial_expression_model_path)
        self.fall_detection_model = load_model(fall_detection_model_path)


    def __del__(self):
        self.__del__()

    def get_facial_expression(self, image, top, bottom, left, right):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[top:bottom, left:right]
        roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                               FACIAL_EXPRESSION_TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine facial expression
        (neural, smile) = self.facial_expression_model.predict(roi)[0]
        return 'Neural' if neural > smile else 'Smile'



