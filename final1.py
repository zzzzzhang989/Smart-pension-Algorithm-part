# 导入包
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
from scipy.spatial import distance as dist
from oldcare.conv import MiniVGGNet
from datetime import datetime

input_video = './tests/activity/desk_01.mp4'

# 全局常量
FACIAL_EXPRESSION_TARGET_WIDTH = 28
FACIAL_EXPRESSION_TARGET_HEIGHT = 28

FALL_DETECTION_TARGET_WIDTH = 64
FALL_DETECTION_TARGET_HEIGHT = 64

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
ACTUAL_DISTANCE_LIMIT = 100  # cm

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

# 控制陌生人检测
strangers_timing = 0  # 计时开始
strangers_start_time = 0  # 开始时间
strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

# 控制微笑检测
facial_expression_timing = 0  # 计时开始
facial_expression_start_time = 0  # 开始时间
facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

# 加载模型

faceutil = FaceUtil(facial_recognition_model_path)
facial_expression_model = MiniVGGNet.build(width=FACIAL_EXPRESSION_TARGET_WIDTH, height=FACIAL_EXPRESSION_TARGET_HEIGHT,
                                           depth=1, classes=2)
facial_expression_model.load_weights(facial_expression_model_path)
fall_detection_model = load_model(fall_detection_model_path)

# 初始化摄像头
if not input_video:
    vs = cv2.VideoCapture(0)
    time.sleep(2)
else:
    vs = cv2.VideoCapture(input_video)

print('[INFO] 开始检测...')
# 不断循环
counter = 0
camera_turned = 0
while True:
    counter += 1
    camera_turned = 0
    (grabbed, image) = vs.read()
    if input_video and not grabbed:
        break

    if not input_video:
        image = cv2.flip(image, 1)

    # time
    datet = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(image, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)  # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 8

    # resize and store gray image to recognize facial expression
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25) # 压缩，加快识别速度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

    # the line to judge whether the camera should rotate
    one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
    five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5),
                               int(VIDEO_HEIGHT / 6 * 5))

    face_location_list, names = faceutil.get_face_location_and_name(image)
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
            rectangle_color = (0    , 255, 0)
        else:
            pass
        cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

        # 陌生人检测逻辑
        if 'Unknown' in names:  # alert
            if strangers_timing == 0:  # just start timing
                strangers_timing = 1
                strangers_start_time = time.time()
            else:  # already started timing
                strangers_end_time = time.time()
                difference = strangers_end_time - strangers_start_time

                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < strangers_limit_time:
                    print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                else:  # strangers appear
                    event_desc = '陌生人出现!!!'
                    event_location = '房间'
                    print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                    cv2.imwrite(os.path.join(output_stranger_path,
                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), image)  # snapshot

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
                        direction = 'left'
                    elif unknown_face_center[0] > five_sixth_image_center[0]:
                        direction = 'right'

                    # adjust to servo
                    if direction:
                        print('%d-摄像头需要 turn %s %d 度' % (counter, direction, ANGLE))

        else:  # everything is ok
            strangers_timing = 0

        if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
            continue

        # 如果检测到有义工存在
        if person_type == 'volunteer':
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
            (neural, smile) = facial_expression_model.predict(roi)[0]
            facial_expression_label = 'Neural' if neural > smile else 'Smile'

            if facial_expression_label == 'Smile':  # alert
                if facial_expression_timing == 0:  # just start timing
                    facial_expression_timing = 1
                    facial_expression_start_time = time.time()
                else:  # already started timing
                    facial_expression_end_time = time.time()
                    difference = facial_expression_end_time - facial_expression_start_time

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    if difference < facial_expression_limit_time:
                        print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
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
                facial_expression_timing = 0

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

    # 义工追踪逻辑
    if 'volunteer' in people_type_list:
        volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
        if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
            print('%d-有义工恰好在可见范围内，摄像头不需要转动' % (counter))
        else:
            adjust_direction = volunteer_adjust_direction_list[0]
            camera_turned = 1
            print('%d-摄像头需要 turn %s %d 度' % (counter, adjust_direction, ANGLE))

    # 在义工和老人之间划线
    if camera_turned == 0:
        for i in volunteer_centroids:
            for j_index, j in enumerate(old_people_centroids):
                pixel_distance = dist.euclidean(i, j)
                face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                actual_distance = pixel_distance / pixel_per_metric

                if actual_distance < ACTUAL_DISTANCE_LIMIT:
                    cv2.line(image, (int(i[0]), int(i[1])),
                             (int(j[0]), int(j[1])), (255, 0, 255), 2)
                    label = 'distance: %dcm' % (actual_distance)
                    cv2.putText(image, label, (image.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    event_desc = '%s正在与义工交互' % (id_card_to_name[old_people_name[j_index]])
                    event_location = '房间桌子'
                    print(
                        '[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (current_time, id_card_to_name[old_people_name[j_index]]))

                    # # store stranger's picture to database
                    # cv2.imwrite(
                    #     os.path.join(output_activity_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                    #     frame)  # snapshot
                    #
                    # # insert the info into database
                    # command = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id %d' % (
                    #     python_path, event_desc, event_location, int(name))
                    # p = subprocess.Popen(command, shell=True)

    # fall detection
    roi = cv2.resize(image, (FALL_DETECTION_TARGET_WIDTH, FALL_DETECTION_TARGET_HEIGHT))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    (fall, normal) = fall_detection_model.predict(roi)[0]
    fall_label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

    # display the label and bounding box rectangle on the output frame
    cv2.putText(image, fall_label, (image.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Fall detection', image)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

vs.release()
cv2.destroyAllWindows()
