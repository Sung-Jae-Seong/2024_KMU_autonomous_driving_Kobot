#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, math
from datetime import datetime

import numpy as np
import cv2, rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan, CompressedImage
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers

import make_log
import pid
import moving_average
import traffic_sign
import ar_tag
import lane_detector
import obstacle_driver
import sensor_driver

# =============================================
# 전역 변수 선언부
# =============================================
motor = None  # 모터 노드 변수
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지
lidar_points = None  # 라이다 데이터를 담을 변수
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
compressed_image = np.empty(shape=[0]) # 압축된 카메라 이미지
motor_msg = xycar_motor()  # 모터 토픽 메시지
ar_msg = {"ID":[],"DX":[],"DZ":[]}  # AR태그 토픽을 담을 변수
CURRENT_ANGLE = 0
lane_detector = lane_detector.LaneDetector()
logger = make_log.Logging()
# 모드 상수 정의
DRIVE = 0
STARTING_LINE = 1
TRAFFIC_SIGN = 2
AR_DRIVE = 3
OBSTACLE_DRIVE = 4
SENSOR_DRIVE = 5
FINISH = 6

# =============================================
# PID 제어기 인스턴스 생성
# =============================================
sensor_pid = pid.PID(kp=2.3, ki=0.035, kd=2.1)
ar_pid = pid.PID(kp=2, ki=0.1, kd=0.1)
pid = pid.PID(kp=0.7, ki=0.02, kd=1.0)

# =============================================
# 콜백 함수들
# =============================================

# USB 카메라 토픽을 받아서 처리하는 콜백함수
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

# USB 카메라 토픽을 받아서 압축된 형태로 저장하는 콜백함수
def compressed_image_callback(data):
    global compressed_image
    np_arr = np.frombuffer(data.data, np.uint8)
    compressed_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# 라이다 토픽을 받아서 처리하는 콜백함수
def lidar_callback(data):
    global lidar_points
    lidar_points = data.ranges

# AR태그 토픽을 받아서 처리하는 콜백함수
def ar_callback(data):
    global ar_msg
    # AR태그의 ID값, X 위치값, Z 위치값을 담을 빈 리스트 준비
    ar_msg = {"ID":[],"DX":[],"DZ":[]}
    # 발견된 모두 AR태그에 대해서 정보 수집하여 ar_msg 리스트에 담음
    for i in data.markers:
        ar_msg["ID"].append(i.id) # AR태그의 ID값을 리스트에 추가
        ar_msg["DX"].append(int(i.pose.pose.position.x*100)) # X값을 cm로 바꿔서 리스트에 추가
        ar_msg["DZ"].append(int(i.pose.pose.position.z*100)) # Z값을 cm로 바꿔서 리스트에 추가

# =============================================
# 모터 토픽 발행 함수
# =============================================
def drive(angle, speed):
    global CURRENT_ANGLE
    angle = int(angle)
    CURRENT_ANGLE = (CURRENT_ANGLE + math.radians(angle)) % (2 * math.pi)
    log_entry = {
        "image": compressed_image,
        "processed": lane_detector.processed_frame,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "position": (
            logger.log_entries[-1]["position"][0] + speed * math.cos(CURRENT_ANGLE),
            logger.log_entries[-1]["position"][1] + speed * math.sin(CURRENT_ANGLE),
        ),
        "direction": CURRENT_ANGLE,
        "speed": speed,
        "lidar": lidar_points,
        "AR_Tag": {"AR_ID": ar_msg["ID"], "AR_DX": ar_msg["DX"], "AR_DZ": ar_msg["DZ"]},
    }
    logger.write_log(log_entry)

    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)

# =============================================
# 차량을 정차시키는 함수
# 지속시간은 0.1초 단위임
# =============================================
def stop_car(duration):
    for _ in range(int(duration)):
        drive(angle=0, speed=0)
        time.sleep(0.1)

# =============================================
# 차량을 이동시키는 함수
# 지속시간은 0.1초 단위임
# =============================================
def move_car(move_angle, move_speed, duration):
    for _ in range(int(duration)):
        drive(move_angle, move_speed)
        time.sleep(0.1)
#=============================================
# 카메라의 Exposure 값을 변경하는 함수 (입력으로 0~255)
#=============================================
def cam_exposure(value):
    command = 'v4l2-ctl -d /dev/videoCAM -c exposure_absolute=' + str(value)
    os.system(command)

#=============================================
# 라바콘 감지
#=============================================
def hsv_rubbercon():
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    upper_orange = np.array([24, 255, 255])
    lower_orange = np.array([0, 70, 115])
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    return cv2.countNonZero(mask)

#=============================================
# 연두색 차량 감지
#=============================================
def hsv_car():
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    upper_orange = np.array([57, 255, 255])
    lower_orange = np.array([42, 91, 81])
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    return cv2.countNonZero(mask)


class Mode:
    def handling_mode(self):
        raise NotImplementedError("handling_mode() should be implemented by subclass")

    def escape_mode(self):
        raise NotImplementedError("escape_mode() should be implemented by subclass")

    def reset_mode(self):
        pass

# 주행 모드 클래스들
class DriveMode(Mode):
    def __init__(self, lane_detector, pid, car_speed, angle_avg):
        self.lane_detector = lane_detector
        self.pid = pid
        self.car_speed = car_speed
        self.angle_avg = angle_avg
        self.prev_car_angle = 0

    def handling_mode(self):
        lane_detected, left_x, right_x = self.lane_detector.detect(image)
        if lane_detected:
            x_midpoint = (left_x + right_x) // 2
            car_angle = (x_midpoint - 240) / 2 # 240은 이미지 중앙
            self.angle_avg.add_sample(car_angle)
            car_angle = self.angle_avg.get_mmed()
            car_angle = self.pid.pid_control(car_angle)
            self.prev_car_angle = car_angle
            drive(car_angle-4, self.car_speed)
        else:
            drive(self.prev_car_angle-4, self.car_speed)

    def escape_mode(self):
        left_points = np.array(lidar_points[130:180])
        valid_points = left_points[np.isfinite(left_points)]
        left_distance = np.min(valid_points) if len(valid_points) > 0 else np.inf

        right_points = np.array(lidar_points[540:590])
        valid_points = right_points[np.isfinite(right_points)]
        right_distance = np.min(valid_points) if len(valid_points) > 0 else np.inf

        middle_points = np.array(lidar_points[0:45] + lidar_points[675:719])
        valid_points = middle_points[np.isfinite(middle_points)]
        middle_distance = np.min(valid_points) if len(valid_points) > 0 else np.inf

        orange_count = hsv_rubbercon()
        green_count = hsv_car()

        if orange_count >= 2000 and left_distance < 0.5 and right_distance < 0.5:
            print("----- Start Sensor Drive... -----")
            return SENSOR_DRIVE
        if green_count >= 1800 and middle_distance < 0.8:
            print("----- Start Obstacle Drive... -----")
            return OBSTACLE_DRIVE
        return None

    def reset_mode(self):
        self.prev_car_angle = 0
        lane_detector.reset()
        pid.reset()
        self.angle_avg.reset()


class StartingLineMode(Mode):
    def __init__(self, ar_detector, ar_pid, basic_speed):
        self.ar_detector = ar_detector
        self.ar_pid = ar_pid
        self.basic_speed = basic_speed
        self.ar_time = None
        self.change_mode = False

    def handling_mode(self):
        self.ar_detector.set_img(image)
        filtered_data = self.ar_detector.filter_ar_data(ar_msg)
        error = self.ar_detector.traffic_ar_drive(filtered_data)
        if error == "is_mask":
            drive(0, 0)
        elif error == "mode_change":
            self.change_mode = True
        control = self.ar_pid.pid_control(error)
        control = max(min(control, 30), -30)
        drive(control, self.basic_speed)

    def escape_mode(self):
        if self.change_mode:
            stop_car(10)
            print("----- Traffic Sign Detecting... -----")
            return TRAFFIC_SIGN
        return None

class TrafficSignMode(Mode):
    def __init__(self, traffic_sign_detector):
        self.traffic_sign_detector = traffic_sign_detector

    def handling_mode(self):
        drive(0, 0)

    def escape_mode(self):
        result = self.traffic_sign_detector.check_traffic_sign(image)
        if result:
            print ("----- AR Following Start... -----")
            return AR_DRIVE
        return None

class ARDriveMode(Mode):
    def __init__(self, ar_detector, ar_pid, basic_speed):
        self.ar_detector = ar_detector
        self.ar_pid = ar_pid
        self.basic_speed = basic_speed
        self.ar_time = None  # 초기값을 None으로 설정
        self.prev_control = 0

    def handling_mode(self):
        filtered_data = self.ar_detector.filter_ar_data(ar_msg)
        error = self.ar_detector.ar_drive(filtered_data)
        if error == "finish":
            self.prev_control *= 0.97
            drive(self.prev_control, self.basic_speed)
            return
        control = self.ar_pid.pid_control(error)
        control = max(min(int(control), 30), -30)
        self.prev_control = control
        drive(control, self.basic_speed)

    def escape_mode(self):
        filtered_data = self.ar_detector.filter_ar_data(ar_msg)
        if filtered_data["ID"] is None:
            if time.time() - self.ar_time > 1.3:
                print("-----------S T A R T  D R I V E-----------")
                return DRIVE
        else:
            self.ar_time = time.time()  # 태그를 찾았을 때 시간을 갱신
        return None

    def reset_mode(self):
        self.ar_time = time.time()
        self.ar_detector.reset()


class ObstacleDriveMode(Mode):
    def __init__(self, obstacle_driver, obstacle_speed):
        self.obstacle_speed = obstacle_speed
        self.obstacle_driver = obstacle_driver

    def handling_mode(self):
        obstacle_lane = self.obstacle_driver.obstacle_lane(lidar_points)
        if obstacle_lane == 'left': # 1차선
            move_car(30, self.obstacle_speed, 7)  # 오
            move_car(-30, self.obstacle_speed, 18)  # 왼
            move_car(40, self.obstacle_speed, 4)  # 오
        elif obstacle_lane == 'right': # 2차선
            move_car(-30, self.obstacle_speed, 8)  # 왼
            move_car(30, self.obstacle_speed, 13)  # 오
            move_car(-40, self.obstacle_speed, 5)  # 왼
        elif obstacle_lane =='self.obstacle_speed': # 3차선
            move_car(-40, self.obstacle_speed, 10)  # 왼
            move_car(40, self.obstacle_speed, 16.5)  # 오
            move_car(-40, self.obstacle_speed, 8.5)  # 왼

    def escape_mode(self):
        print("----- Return to Drive... -----")
        return DRIVE


class SensorDriveMode(Mode):
    def __init__(self, sensor_driver, sensor_pid, lane_detector, sensor_speed):
        self.sensor_driver = sensor_driver
        self.sensor_pid = sensor_pid
        self.lane_detector = lane_detector
        self.sensor_time = time.time()
        self.sensor_speed = sensor_speed

    def handling_mode(self):
        error = self.sensor_driver.sensor_error(lidar_points)
        sensor_angle = 10*(self.sensor_pid.pid_control(error))
        drive(sensor_angle, self.sensor_speed)

    def escape_mode(self):
        orange_count = hsv_rubbercon()
        if orange_count <= 300:
            if time.time() - self.sensor_time > 0.3:
                print("----- Return to Drive... -----")
                return DRIVE
        else:
            self.sensor_time = time.time()
        return None

    def reset_mode(self):
        self.sensor_pid.reset()
        self.sensor_time = time.time()


class FinishMode(Mode):
    def handling_mode(self):
        cv2.destroyAllWindows()
        stop_car(10)
        print("----- Bye~! -----")
        logger.save_logs_to_file()

    def escape_mode(self):
        return None


def init_topic():
    global motor
    cam_exposure(100)  # 카메라의 Exposure 값을 변경
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, compressed_image_callback, queue_size=1)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback, queue_size=1 )
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    #=========================================
    # 발행자 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready ----------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    rospy.wait_for_message("ar_pose_marker", AlvarMarkers)
    print("AR detector Ready ----------")

    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")
    # 일단 차량이 움직이지 않도록 정지상태로 만듭니다.
    stop_car(10) # 1초 동안 정차


#=============================================
# 실질적인 메인 함수
#=============================================
def start():
    init_topic()
    #=========================================
    # 메인 루프
    #=========================================
    traffic_sign_detector = traffic_sign.TrafficSignDetector()
    ar_detector = ar_tag.AR()
    angle_avg_count = 6
    angle_avg = moving_average.MovingAverage(angle_avg_count)
    obstacleDriver = obstacle_driver.ObstacleDrive()
    sensorDriver = sensor_driver.SensorDrive()

    move_car(0, 0, 1)

    ar_speed = 5
    drive_speed = 11
    obstacle_speed = 7
    sensor_speed = 10

    # 각 모드와 클래스 인스턴스를 매핑합니다.
    modes = {
        DRIVE: DriveMode(lane_detector, pid, drive_speed, angle_avg),
        STARTING_LINE: StartingLineMode(ar_detector, ar_pid, ar_speed),
        TRAFFIC_SIGN: TrafficSignMode(traffic_sign_detector),
        AR_DRIVE: ARDriveMode(ar_detector, ar_pid, ar_speed),
        OBSTACLE_DRIVE: ObstacleDriveMode(obstacleDriver, obstacle_speed),
        SENSOR_DRIVE: SensorDriveMode(sensorDriver, sensor_pid, lane_detector, sensor_speed),
        FINISH: FinishMode(),
    }

    drive_mode = STARTING_LINE  # 초기 모드를 설정합니다.
    current_mode = modes[drive_mode]

    while not rospy.is_shutdown():
        current_mode.handling_mode()
        next_mode = current_mode.escape_mode()
        if next_mode is not None:
            current_mode = modes[next_mode]
            current_mode.reset_mode()


#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
