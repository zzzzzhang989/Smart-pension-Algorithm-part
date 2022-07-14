
# this is used to yard
import argparse
from datetime import datetime
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


graph = tf.get_default_graph()

input_video = './tests/corridor_01.avi'

# 全局常量
FACIAL_EXPRESSION_TARGET_WIDTH = 28
FACIAL_EXPRESSION_TARGET_HEIGHT = 28

FALL_DETECTION_TARGET_WIDTH = 64
FALL_DETECTION_TARGET_HEIGHT = 64

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20

people_info_path = 'info/people_info.csv'
facial_expression_info_path = 'info/facial_expression_info.csv'
facial_recognition_model_path = 'models/face_recognition_hog.pickle'
facial_expression_model_path = 'models/face_expression.hdf5'
fall_detection_model_path = 'models/fall_detection.hdf5'
output_stranger_path = 'supervision/strangers'
output_smile_path = 'supervision/smile'
python_path = '/home/reed/anaconda3/envs/tensorflow/bin/python'

# 得到 ID->姓名的map 、 ID->职位类型的map、
# 摄像头ID->摄像头名字的map、表情ID->表情名字的map
id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)
facial_expression_id_to_name = fileassistant.get_facial_expression_info(facial_expression_info_path)

# 加载模型

faceutil = FaceUtil(facial_recognition_model_path)
facial_expression_model = MiniVGGNet.build(width=FACIAL_EXPRESSION_TARGET_WIDTH, height=FACIAL_EXPRESSION_TARGET_HEIGHT,
                                           depth=1, classes=2)
facial_expression_model.load_weights(facial_expression_model_path)
fall_detection_model = load_model(fall_detection_model_path)


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture("rtmp://39.107.230.98:1935/live/home")
        # 不断循环
        self.counter = 0

        # 控制陌生人检测
        self.strangers_timing = 0  # 计时开始
        self.strangers_start_time = 0  # 开始时间
        self.strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

        # 控制微笑检测
        self.facial_expression_timing = 0  # 计时开始
        self.facial_expression_start_time = 0  # 开始时间
        self.facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling


    def __del__(self):
        self.cap.release()

    # 处理画面（画警戒区域等等）
    def get_frame(self):
        self.counter = self.counter +1
        success, image = self.cap.read()

        if not success:
            return
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)


        # 标注左上角的时间
        # datet = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # cv2.putText(image, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (255, 255, 255), 2, cv2.LINE_AA)  # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 8


        # 算法部分
        # image = imutils.resize(image, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)  # 压缩，加快识别速度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

        face_location_list, names = faceutil.get_face_location_and_name(image)

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
        five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5),
                                   int(VIDEO_HEIGHT / 6 * 5))

        cv2.line(image, (one_sixth_image_center[0], 0),
                 (one_sixth_image_center[0], VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(image, (five_sixth_image_center[0], 0),
                 (five_sixth_image_center[0], VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        people_type_list = list(set([id_card_to_type[i] for i in names]))

        volunteer_name_direction_dict = {}
        volunteer_centroids = []
        old_people_centroids = []
        old_people_name = []


        # 处理每一张识别到的人脸
        for ((left, top, right, bottom), name) in zip(face_location_list, names):
            person_type = id_card_to_type[name]
            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if person_type == 'old_people':
                rectangle_color = (0, 0, 128)
            elif person_type == 'employee':
                rectangle_color = (255, 0, 0)
            elif person_type == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass

            # 陌生人检测逻辑
            if 'Unknown' in names:  # alert
                if self.strangers_timing == 0:  # just start timing
                    self.strangers_timing = 1
                    self.strangers_start_time = time.time()
                else:  # already started timing
                    self.strangers_end_time = time.time()
                    difference = self.strangers_end_time - self.strangers_start_time

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))

                    if difference < self.strangers_limit_time:
                        print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                    else:  # strangers appear
                        event_desc = '陌生人出现!!!'
                        event_location = '房间'
                        print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                        cv2.imwrite(os.path.join(output_stranger_path,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    image)  # snapshot

                        # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s' % (
                        #     python_path, event_desc, event_location)
                        # p = subprocess.Popen(command, shell=True)

                        # 开始陌生人追踪
                        unknown_face_center = (int((right + left) / 2), int((top + bottom) / 2))

                        cv2.circle(image, (unknown_face_center[0], unknown_face_center[1]), 4, (0, 255, 0), -1)

                        direction = ''
                        # face locates too left, servo need to turn right,
                        # so that face turn right as well
                        if unknown_face_center[0] < one_sixth_image_center[0]:
                            direction = 'right'
                        elif unknown_face_center[0] > five_sixth_image_center[0]:
                            direction = 'left'

                        # adjust to servo
                        if direction:
                            print('%d-摄像头需要 turn %s %d 度' % (self.counter, direction, ANGLE))
            else:  # everything is ok
               self.strangers_timing = 0

            cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

            # if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
            #     continue

            if person_type == 'volunteer':  # 如果检测到有义工存在
                # 获得义工位置
                volunteer_face_center = (int((right + left) / 2),
                                         int((top + bottom) / 2))
                volunteer_centroids.append(volunteer_face_center)

                cv2.circle(image,
                           (volunteer_face_center[0], volunteer_face_center[1]),
                           8, (255, 0, 0), -1)

                adjust_direction = ''
                # face locates too left, servo need to turn right,
                # so that face turn right as well
                if volunteer_face_center[0] < one_sixth_image_center[0]:
                    adjust_direction = 'right'
                elif volunteer_face_center[0] > five_sixth_image_center[0]:
                    adjust_direction = 'left'

                volunteer_name_direction_dict[name] = adjust_direction

            elif person_type == 'old_people':  # 如果没有发现义工
                old_people_face_center = (int((right + left) / 2),
                                          int((top + bottom) / 2))
                old_people_centroids.append(old_people_face_center)
                old_people_name.append(name)

                cv2.circle(image,
                           (old_people_face_center[0], old_people_face_center[1]),
                           4, (0, 255, 0), -1)
            else:
                pass

            # 表情检测逻辑
            # 如果不是陌生人，且对象是老人
            if name != 'Unknown' and id_card_to_type[name] == 'old_people':
                # 表情检测逻辑
                roi = gray[top:bottom, left:right]
                roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                                       FACIAL_EXPRESSION_TARGET_HEIGHT))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # determine facial expression
                # (neural, smile) = facial_expression_model.predict(roi)[0]
                global graph
                with graph.as_default():
                    (neural, smile) = facial_expression_model.predict(roi)[0]
                facial_expression_label = 'Neural' if neural > smile else 'Smile'

                if facial_expression_label == 'Smile':  # alert
                    if self.facial_expression_timing == 0:  # just start timing
                        self.facial_expression_timing = 1
                        self.facial_expression_start_time = time.time()
                    else:  # already started timing
                        self.facial_expression_end_time = time.time()
                        difference = self.facial_expression_end_time - self.facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < self.facial_expression_limit_time:
                            print(
                                '[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在笑' % (id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                            cv2.imwrite(os.path.join(output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        image)  # snapshot

                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                                python_path, event_desc, event_location, int(name))
                            p = subprocess.Popen(command, shell=True)

                else:  # everything is ok
                    self.facial_expression_timing = 0

            else:  # 如果是陌生人，则不检测表情
                facial_expression_label = ''

                # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            final_label = id_card_to_name[name]
            draw.text((left, top - 30), final_label + facial_expression_label,
                      font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                      fill=(255, 0, 0))  # linux
            # 转换回OpenCV格式
            image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # fall detection
        roi = cv2.resize(image, (FALL_DETECTION_TARGET_WIDTH, FALL_DETECTION_TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # (fall, normal) = fall_detection_model.predict(roi)[0]
        # global graph
        with graph.as_default():
            (fall, normal) = fall_detection_model.predict(roi)[0]
        fall_label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, fall_label, (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)







        # show our detected faces along with smiling/not smiling labels
        ret, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()
        # else:
        #     print("wrong read")
