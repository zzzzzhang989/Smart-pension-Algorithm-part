import dlib
import cv2
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from scipy.spatial import distance as dist
from PIL import Image, ImageDraw, ImageFont
from keras.preprocessing.image import img_to_array

from oldcare.utils import fileassistant
from oldcare.conv import MiniVGGNet
from oldcare.track import CentroidTracker
from oldcare.track import TrackableObject
from oldcare.facial import FaceUtil

graph = tf.get_default_graph()

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

SKIP_FRAMES = 30
MINIMUM_CONFIDENCE = 0.95

people_info_path = 'info/people_info.csv'
facial_expression_info_path = 'info/facial_expression_info.csv'
facial_expression_model_path = 'models/face_expression.hdf5'
facial_recognition_model_path = 'models/face_recognition_hog.pickle'
fall_detection_model_path = 'models/fall_detection.hdf5'
object_recognize_model_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
prototxt_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'

# 得到 ID->姓名的map 、 ID->职位类型的map、
# 摄像头ID->摄像头名字的map、表情ID->表情名字的map
id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)
facial_expression_id_to_name = fileassistant.get_facial_expression_info(facial_expression_info_path)

# 加载模型

face_recognize_model = FaceUtil(facial_recognition_model_path)
facial_expression_model = MiniVGGNet.build(width=FACIAL_EXPRESSION_TARGET_WIDTH, height=FACIAL_EXPRESSION_TARGET_HEIGHT,
                                           depth=1, classes=2)
facial_expression_model.load_weights(facial_expression_model_path)
fall_detection_model = load_model(fall_detection_model_path)
object_recognize_model = cv2.dnn.readNetFromCaffe(prototxt_file_path, object_recognize_model_path)

# 物体识别模型能识别的物体（21种）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture("rtmp://39.107.230.98:1935/live/home")

        # 不断循环
        self.counter = 0
        self.camera_turned = 0

        # 控制陌生人检测
        self.strangers_timing = 0  # 计时开始
        self.strangers_start_time = 0  # 开始时间
        self.strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

        # 控制微笑检测
        self.facial_expression_timing = 0  # 计时开始
        self.facial_expression_start_time = 0  # 开始时间
        self.facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

        self.one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
        self.five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5), int(VIDEO_HEIGHT / 6 * 5))

        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.trackers = []
        self.trackableObjects = {}

    def __del__(self):
        self.cap.release()

    # 0 means the room camera, 1 means the yard camera
    def get_frame(self, type):
        self.counter = self.counter + 1
        self.camera_turned = 0
        success, image = self.cap.read()

        if not success:
            return

        image = cv2.resize(image, (VIDEO_WIDTH, VIDEO_HEIGHT))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

        # 标注左上角的时间
        datet = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(image, datet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)  # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 8

        # the line to judge whether the camera should rotate
        one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
        five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5), int(VIDEO_HEIGHT / 6 * 5))

        # 算法部分

        # invading detect
        if type == 1:
            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if self.totalFrames % SKIP_FRAMES == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(image, 0.007843, (VIDEO_WIDTH, VIDEO_HEIGHT), 127.5)
                object_recognize_model.setInput(blob)
                detections = object_recognize_model.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > MINIMUM_CONFIDENCE:
                        # extract the index of the class label from the
                        # detections list
                        idx = int(detections[0, 0, i, 1])

                        # if the class label is not a person, ignore it
                        if CLASSES[idx] != "person":
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = detections[0, 0, i, 3:7] * np.array(
                            [VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT])
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
                for tracker in self.trackers:
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
            cv2.line(image, (0, VIDEO_HEIGHT // 2), (VIDEO_WIDTH, VIDEO_HEIGHT // 2), (0, 255, 255), 2)

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = self.trackableObjects.get(objectID, None)

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
                        if direction < 0 and centroid[1] < VIDEO_HEIGHT // 2:
                            self.totalUp += 1
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > VIDEO_HEIGHT // 2:
                            self.totalDown += 1
                            to.counted = True

                            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
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
                self.trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame

                text = "ID {}".format(objectID)
                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                # ("Up", totalUp),
                ("Down", self.totalDown),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image, text, (10, VIDEO_HEIGHT - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        face_location_list, names = face_recognize_model.get_face_location_and_name(image)
        people_type_list = list(set([id_card_to_type[i] for i in names]))

        volunteer_name_direction_dict = {}
        volunteer_centroids = []
        old_people_centroids = []
        old_people_name = []

        # loop over the face bounding boxes
        for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人
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
                        print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))

                        # 开始陌生人追踪
                        if type == 1:
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

            # 表情检测逻辑
            # 如果不是陌生人，且对象是老人,and in room
            if type == 0 and name != 'Unknown' and id_card_to_type[name] == 'old_people':
                # 表情检测逻辑
                roi = gray[top:bottom, left:right]
                roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                                       FACIAL_EXPRESSION_TARGET_HEIGHT))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # determine facial expression
                global graph
                with graph.as_default():
                    (neural, smile) = facial_expression_model.predict(roi)[0]
                facial_expression_label = 'Neural' if neural > smile else 'Smile'

                if facial_expression_label == 'Smile':  # alert
                    if self.facial_expression_timing == 0:  # just start timing
                        self.facial_expression_timing = 1
                        self.facial_expression_start_time = time.time()
                    else:  # already started timing
                        difference = time.time() - self.facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < self.facial_expression_limit_time:
                            print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (
                                current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))

                            # cv2.imwrite(os.path.join(output_smile_path,
                            #                          'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            #             image)  # snapshot


                else:  # everything is ok
                    self.facial_expression_timing = 0

            else:  # 如果是陌生人，则不检测表情
                facial_expression_label = ''

            if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
                continue

            # 如果检测到有义工存在
            if person_type == 'volunteer' and type == 0:
                # 获得义工位置
                volunteer_face_center = (int((right + left) / 2),
                                         int((top + bottom) / 2))
                volunteer_centroids.append(volunteer_face_center)

                cv2.circle(image, (volunteer_face_center[0], volunteer_face_center[1]), 8, (255, 0, 0), -1)

                adjust_direction = ''
                # face locates too left, servo need to turn right,
                # so that face turn right as well
                if volunteer_face_center[0] < one_sixth_image_center[0]:
                    adjust_direction = 'right'
                elif volunteer_face_center[0] > five_sixth_image_center[0]:
                    adjust_direction = 'left'

                volunteer_name_direction_dict[name] = adjust_direction

            elif person_type == 'old_people' and type == 0:  # 如果没有发现义工
                old_people_face_center = (int((right + left) / 2),
                                          int((top + bottom) / 2))
                old_people_centroids.append(old_people_face_center)
                old_people_name.append(name)

                cv2.circle(image, (old_people_face_center[0], old_people_face_center[1]), 4, (0, 255, 0), -1)
            else:
                pass

            # 人脸识别，把人名写上 (处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            final_label = id_card_to_name[name]
            draw.text((left, top - 30), final_label, font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                      fill=(255, 0, 0))
            # 转换回OpenCV格式
            image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # 义工追踪逻辑
        if 'volunteer' in people_type_list and type == 0:
            volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
            if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
                print('%d-有义工恰好在可见范围内，摄像头不需要转动' % self.counter)
            else:
                adjust_direction = volunteer_adjust_direction_list[0]
                self.camera_turned = 1
                print('%d-摄像头需要 turn %s %d 度' % (self.counter, adjust_direction, ANGLE))

        # activity detect
        if self.camera_turned == 0 and type == 0:
            for i in volunteer_centroids:
                for j_index, j in enumerate(old_people_centroids):
                    pixel_distance = dist.euclidean(i, j)
                    face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                    pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                    actual_distance = pixel_distance / pixel_per_metric

                    if actual_distance < ACTUAL_DISTANCE_LIMIT:
                        cv2.line(image, (int(i[0]), int(i[1])),
                                 (int(j[0]), int(j[1])), (255, 0, 255), 2)
                        label = 'distance: %dcm' % actual_distance
                        cv2.putText(image, label, (image.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        event_desc = '%s正在与义工交互' % (id_card_to_name[old_people_name[j_index]])
                        event_location = '房间桌子'

                        print(
                            '[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (current_time, id_card_to_name[old_people_name[j_index]]))

                        # cv2.imwrite(
                        #     os.path.join(output_activity_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                        #     frame)  # snapshot

                        # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id %d' % (
                        #     python_path, event_desc, event_location, int(name))
                        # p = subprocess.Popen(command, shell=True)

        # fall detect
        roi = cv2.resize(gray, (FALL_DETECTION_TARGET_WIDTH, FALL_DETECTION_TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        global graph
        with graph.as_default():
            (fall, normal) = fall_detection_model.predict(roi)[0]
        fall_label = "Fall (%.2f)" % fall if fall > normal else "Normal (%.2f)" % (normal)

        # here to insert into database or send to front end

        # show our detected faces along with smiling/not smiling labels
        ret, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()
        # else:
        #     print("wrong read")
