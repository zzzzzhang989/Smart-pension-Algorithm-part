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
from oldcare.track import CentroidTracker
from oldcare.track import TrackableObject
from datetime import datetime
import dlib

input_video = './tests/invade/yard_01.mp4'

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

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
# 超参数
# minimum probability to filter weak detections
minimum_confidence = 0.80

# 物体识别模型能识别的物体（21种）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

prototxt_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
people_info_path = 'info/people_info.csv'
facial_recognition_model_path = 'models/face_recognition_hog.pickle'
fall_detection_model_path = 'models/fall_detection.hdf5'
output_stranger_path = 'supervision/strangers'
output_smile_path = 'supervision/smile'
python_path = '/home/reed/anaconda3/envs/tensorflow/bin/python'

# 得到 ID->姓名的map 、 ID->职位类型的map、
# 摄像头ID->摄像头名字的map、表情ID->表情名字的map
id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)

# 控制陌生人检测
strangers_timing = 0  # 计时开始
strangers_start_time = 0  # 开始时间
strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

skip_frames = 30  # of skip frames between detections
# 加载模型
faceutil = FaceUtil(facial_recognition_model_path)
fall_detection_model = load_model(fall_detection_model_path)
object_recognize_model = cv2.dnn.readNetFromCaffe(prototxt_file_path, model_file_path)

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
totalFrames = 0
totalDown = 0
totalUp = 0

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
    image = cv2.resize(image, (500, 500))  # 压缩，加快识别速度

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = image.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)
        object_recognize_model.setInput(blob)
        detections = object_recognize_model.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > minimum_confidence:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # draw a rectangle around the people
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    event_desc = '有人闯入禁止区域!!!'
                    event_location = '院子'
                    print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % (current_time))

                    # cv2.imwrite(
                    #     os.path.join(output_fence_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                    #     frame)  # snapshot
                    #
                    # # insert into database
                    # command = '%s inserting.py --event_desc %s --event_type 4 --event_location %s' % (
                    #     python_path, event_desc, event_location)
                    # p = subprocess.Popen(command, shell=True)

                # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4,
                   (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        # ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
            rectangle_color = (0, 255, 0)
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
                        direction = 'right'
                    elif unknown_face_center[0] > five_sixth_image_center[0]:
                        direction = 'left'

                    # adjust to servo
                    if direction:
                        print('%d-摄像头需要 turn %s %d 度' % (counter, direction, ANGLE))

        else:  # everything is ok
            strangers_timing = 0

        if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
            continue

        # 人脸识别结束后，把人名写上 (处理中文显示问题)
        img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        final_label = id_card_to_name[name]
        draw.text((left, top - 30), final_label, font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40), fill=(255, 0, 0))
        # 转换回OpenCV格式
        image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

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

    totalFrames += 1

    cv2.imshow('Fall detection', image)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

vs.release()
cv2.destroyAllWindows()
