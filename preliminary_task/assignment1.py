#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : parking.py
# 코드작성팀명 : Kobot
####################################################################

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import math
import heapq
import numpy as np
import pygame

import rospy
from xycar_msgs.msg import xycar_motor

#=============================================
# 모터 토픽을 발행할 것임을 선언
#=============================================
motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
xycar_msg = xycar_motor()

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
AR = (1142, 62) # AR 태그의 위치
P_ENTRY = (1036, 162) # 주차라인 진입 시점의 좌표
P_END = (1129, 69) # 주차라인 끝의 좌표
MAP_X0, MAP_Y0 = 0, 0 #맵의 시작 좌표
MAP_X1, MAP_Y1 = 0, 0 #맵의 끝 좌표
#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    xycar_msg.angle = int(angle)
    xycar_msg.speed = int(speed)
    motor_pub.publish(xycar_msg)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
rx, ry = [], []

R = 461.5773942453219/2 # 차량이 최대 조향각으로 그리는 원의 반지름
# 반지름 R 계산과정 ----------------------------------
# tracking 함수에 print(x, y)를 추가한다.
# 차량이 최대 조향각 angle=50, speed=50으로 움직이는 좌표를 저장한다.
# 저장 명령어 : roslaunch assignment_1 parking.launch > position.txt
# 가장 큰 x좌표에서 가장 작은 x좌표를 빼면 원의 지름을 구할 수 있다.
# max(x) - min(x) = 2R을 통해 반지름을 구한다.

SINGLE_STEP = 2 # 노드를 탐색할 단위 STEP

#=============================================
# 경로를 생성하는 함수
# 차량의 시작위치 sx, sy, 시작각도 syaw
# 최대가속도 max_acceleration, 단위시간 dt 를 전달받고
# 경로를 리스트를 생성하여 반환한다.
#=============================================

## planing ###
"""
### Astar 알고리즘을 사용하여 목적지까지의 최단거리를 찾는다. ###

### 기존 Astar 알고리즘과 현재 시스템의 차이 ###
1. Astar는 그리드 기반이지만, 현재 시스템은 연속적이다.
2. Astar는 장애물과 맵이 고정되어 있지만, 현재 시스템은 차량의 회전 반경을 고려해서 경로를 생성해야 한다.

### 장애물과 경로 설정 ###
1. 차량이 최대 조향각으로 설정되어 있을 때 차량이 가지 못하는 영역을 장애물로 간주한다.
2. Astar는 한 번 탐색한 노드를 다시 방문하지 않지만, 현재 시스템에서는 차량의 진행 방향을 고려해서 중복 방문을 허용한다.
   (만약 방문한 노드를 다시 방문했을 때 목적지와의 방향이 일치한다면 그 노드를 무시하지 않고 path에 추가한다)
3. 이동비용(g)값에 추가 가중치를 적용하고 possibility라는 항목을 추가해서 구현한다.

### 선형보간법 사용 ###
    경로 사이에 추가 포인트을 넣어 경로를 더 부드럽게 만든다.
"""

def syaw_to_radian(syaw):
# syaw를 라디안으로 변경하는 함수
# planning의 syaw, tracking의 yaw가 표현방법이 달라 변경이 필요하다.
    converted_syaw = (360 + ((360-syaw) - 90))%360
    return math.radians(converted_syaw)

def distance(pos1, pos2):
# 두 점 사이의 거리를 계산한다
    return math.dist(pos1, pos2)

def find_circle_centers(pos, d, r=R):
# 원의 좌표를 계산하는 함수
# ----------------------------------------------------
# 1. 좌표와 해당 좌표의 기울기를 입력받고
# 2. 해당 직선에서 접하는 2개의 원의 중심 좌표를 반환한다
# ----------------------------------------------------
    #기울기의 범위를 확인하고 수직, 수평선으로 처리한다.
    if abs(d) > MAP_Y1:
        d = float('inf')
    elif abs(d) < 1/MAP_X1:
        d = 0

    # 기울기가 무한대이면 x값만 +-r을 해서 원의 좌표 업데이트한다.
    if d == float('inf'):  # 기울기가 무한대인 경우, 수직선 처리한다.
        return [(pos[0]+r, pos[1]), (pos[0]-r, pos[1])]
    # 기울기가 0이면 y값만 +-r을 해서 원의 좌표 업데이트한다.
    if d == 0:
        return [(pos[0], pos[1]+r), (pos[0], pos[1]-r)]
    # 일반적인 경우
    # 기울기 d에 대한 수직 벡터
    dx, dy = -1, d
    len_d = math.sqrt(dx**2 + dy**2)
    dx, dy = r * dx / len_d, r * dy / len_d  # 단위 벡터로 변환하고 반지름을 곱한다.
    # 원의 중심 좌표 계산한다.
    x1, y1 = pos[0] + dx, pos[1] + dy
    x2, y2 = pos[0] - dx, pos[1] - dy
    return [(x1, y1), (x2, y2)]


def update_obstacles(new_node):
# 새로운 node에 대해서 장애물을 업데이트 하는 함수
    # 만약 새로운 노드와 그 부모 노드가 연속으로 원의 바깥에 있고,
    # 직선을 그리며 빠져나가면 새로운 장애물을 업데이트 해야한다.
    if (new_node.parent.incircle == new_node.incircle == False
        and new_node.parent.step == new_node.step):
        # 새로운 노드의 좌표와 기울기를 통해 새로운 장애물의 좌표를 구한다
        if new_node.step[0] == 0: # 만약 수직으로 빠져나가면 기울기에 무한대를 입력
            return find_circle_centers(new_node.position, float('inf'))
        return find_circle_centers(new_node.position, (new_node.step[1]/new_node.step[0]))
    # 아직 기존의 원(장애물)을 벗어나지 못했다고 판단되면 기존의 원의 좌표를 반환한다.
    return new_node.parent.obstacles

def find_slope_intercept(pos1, pos2):
# 두 점 사이의 기울기와 y절편을 계산하는 함수
    if pos1[0] == pos2[0]:
        return float('inf'), pos1[1]  # y절편은 이 경우 수직선의 x 위치다.
    m = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
    b = pos1[1] - m * pos1[0]
    return m, b

def update_possibility(new_node, endNode):
# 목적지까지 직선으로 갈 수 있는 노드를 검사하는 함수
# 조건 : 현재 노드부터 목적지까지의 기울기가 -1일 때 True
# 이유 : P_ENTRY부터 P_END까지의 기울기는 -1이므로
# ----------------------------------------------------
# 1. 노드로부터 목적지까지 직선의 정보를 구한다
# 2. 기울기가 -1이 아니라면 False
# 3. 노드부터 P_END까지의 직선이 장애물 안에 포함되어 있으면 False
# 4. 나머지의 경우 True
# ----------------------------------------------------
    # 노드부터 목적지까지의 직선의 정보를 구한다.
    derv, point_x = find_slope_intercept(new_node.position, endNode.position)
    if derv != -1:
        return False
    #점과 직선 사이의 거리를 계산하기 위한 분모 값
    sqrt_denom = math.sqrt(derv ** 2 + 1)
    for obs in new_node.obstacles:
        #만약 직선부터 장애물 좌표까지의 거리가 R-SINGLE_STEP/2보다 크다면
        #장애물 바깥에 있는 것이므로 True 아니라면 False
        distance = abs(derv * obs[0] - obs[1] + point_x) / sqrt_denom
        if distance < R-SINGLE_STEP/2:
            return False
    return True


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        self.step = ()
        self.incircle = False
        self.obstacles = []
        self.possibility = False
    # 사용의 편의를 위해 __eq__를 오버라이딩한다.
    # == 연산 시 position을 기준으로 판단한다.
    def __eq__(self, other):
        return self.position == other.position
    #사용의 편의를 위해 <를 오버라이딩한다.
    # < 연산 사용시 self.f를 기준으로 판단
    # 추후 노드가 heapq에 삽입될 때 f값을 기준으로 힙 자료구조의 적절한 위치에 삽입된다.
    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal, D=1, D2=2 ** 0.5):
# Astar알고리즘의 휴리스틱 함수를 정의
# D * (dx + dy)는 맨해튼 거리이다. 수평 및 수직 이동을 고려하여 거리를 계산한다.
# (D2 - 2*D) * min(dx, dy) 대각선 이동을 고려하는 부분(맨해튼 거리 보완)
# 유클리드 거리를 이용하면 격자점을 통한 이동을 고려하기 힘든 반면 맨해튼 거리는 이것을 더 잘 표현할 수 있다.
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def aStar(start, end, start_d, end_d):
# Astar 알고리즘
# 시작 좌표와, 목적지 좌표, 시작 좌표에서의 장애물 좌표를 입력받는다.
# ----------------------------------------------------
# 1. 시작 노드와 목적지 노드를 초기화한다.
# 2. 탐색 리스트와 탐색완료목록을 초기화한다.
# 3. 목적지에 도달할 때까지 노드를 탐색한다.
# 4. 생성된 이웃 노드가 맵 바깥에 위치하거나, 장애물 내부에 위치하는 경우 해당 노드의 탐색을 중지한다.
# 5. 장애물정보, 가능성, 이동 비용(g), 휴리스틱 값(h), f값(f = g + h)을 업데이트한다.
# ----------------------------------------------------
    #startNode와 endNode 초기화
    startNode = Node(None, start)
    startNode.obstacles = find_circle_centers(startNode.position, start_d)
    endNode = Node(None, end)
    #목적지에 -1기울기로 들어와야 하므로 목적지의 좌표와 기울기를 입력하여 목적지에서의 장애물 좌표를 구한다.
    endNode.obstacles = find_circle_centers(endNode.position, end_d)
    #탐색할 노드
    openList = []
    #이미 탐색한 노드
    closedDict = {}
    heapq.heappush(openList, startNode)
    while openList:
        currentNode = heapq.heappop(openList)
        closedDict[currentNode.position] = currentNode
        #만약 현재 Node가 endNode의 좌표와 같다면 경로 반환
        if currentNode == endNode:
            path = []
            while currentNode:
                path.append(currentNode.position)
                currentNode = currentNode.parent
            return path[::-1]
        #다음 step을 구하고 탐색할 노드 생성하는 단계
        steps = [
        	(0, -SINGLE_STEP), (0, SINGLE_STEP),
        	(-SINGLE_STEP, 0), (SINGLE_STEP, 0),
        	(SINGLE_STEP/2,SINGLE_STEP/2),(-SINGLE_STEP/2,SINGLE_STEP/2),
        	(SINGLE_STEP/2,-SINGLE_STEP/2), (-SINGLE_STEP/2,-SINGLE_STEP/2)
        	]
        for step in steps:
            step = (step[0], -1*step[1]) #pygame에서는 y축의 방향이 반대인것을 고려
            nodePosition = (currentNode.position[0] + step[0], currentNode.position[1] + step[1])
            #nodePosition이 맵의 바깥에 위치하면 탐색 중지
            if (nodePosition[0] < MAP_X0 or nodePosition[1] < MAP_Y0
            	or nodePosition[0] > MAP_X1 or nodePosition[1] > MAP_Y1):
                continue
            new_node = Node(currentNode, nodePosition)
            #다음 노드의 포지션이 현재 장애물 안에 있으면 탐색 중지
            min_distance = min(
	            distance(new_node.position, obstacle) for obstacle in currentNode.obstacles
	            )
            #연속적인 맵에 대해서 불연속적인 step으로 경로를 탐색하기 때문에 step의 1/2만큼의 오차를 허용하여 근사한다.
            if min_distance <= R-SINGLE_STEP/2:
                continue
            #new node 정보 업데이트
            new_node.step = step
            new_node.obstacles = update_obstacles(new_node)
            new_node.possibility = update_possibility(new_node, endNode)
            #중복 제거를 위한 코드
            #이미 탐색한 노드와 겹칠 때
            if new_node.position in closedDict:
                #새로운 노드는 목적지까지 직선으로 갈 수 있고
                #기존의 노드는 직선으로 못간다면 closedDict를 새로운 노드로 업데이트 한다
                if new_node.possibility and not closedDict[new_node.position].possibility:
                    closedDict[new_node.position]=new_node
                #그렇지 않으면 탐색을 중지한다.
                else:
                    continue
            if min_distance <= R+SINGLE_STEP/2:
                new_node.incircle=True
            #노드의 g, h, f값을 업데이트한다.
            g_weight = 1
            new_node.g = currentNode.g + g_weight
            #만약 새로운 노드가 endNode의 장애물 안에 있다면 g값에 비용을 추가해서
            #이러한 노드를 피하는 경로를 선호하게 만든다.
            if min(
                    distance(new_node.position, obstacle) for obstacle in endNode.obstacles
                ) <= R-SINGLE_STEP/2:
                #endNode의 장애물로 진행하는 1회의 step과 그렇지 않은 3회의 step에 대해 똑같은 g값을 갖게하여
                #endNode에 진입할 때 매끄럽게 진입하도록 유도한다.
                new_node.g += g_weight*2
            new_node.h = heuristic(new_node, endNode)
            new_node.f = new_node.g + new_node.h
            #만약 openList에서 new_node와 같은 위치의 노드가 있고 그 노드의 g비용이 new_node와 같거나 작으면 탐색을 중지한다.
            if any(child for child in openList if new_node == child and new_node.g >= child.g):
                continue
            heapq.heappush(openList, new_node)
    return None

def interpolate_path(rx, ry):
# 주어진 경로 (rx, ry)에 대해 선형보간을 수행하는 함수
# 경로 사이에 추가 포인트을 넣어 경로를 더 부드럽게 만든다.
# --------------------------------------
# 1. 입력 리스트의 길이를 확인하여 보간의 필요 여부를 판단한다.
# 2. 필요할 경우 선형 보간 알고리즘을 통해 새로운 포인트을 추가한다.
# --------------------------------------
    # 입력받은 rx 및 ry 리스트의 길이(n)를 확인한다.
    n = len(rx)
    if n <= 1:
        return rx, ry  # 길이가 1 이하면 보간이 필요없다고 판단하여 그대로 반환한다.
    # 보간된 결과를 저장할 리스트
    rx_interpolated = []
    ry_interpolated = []
    # 선형 보간을 통해 원래의 포인트 사이에 새로운 포인트 추가한다.
    for i in range(n - 1):
        # 현재 포인트(x1, y1)와 다음 포인트(x2, y2)
        x1, y1 = rx[i], ry[i]
        x2, y2 = rx[i + 1], ry[i + 1]
        # 중간 포인트(xm, ym) 계산한다.
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        # 중간 포인트와 현재 포인트를 동시에 추가한다.
        rx_interpolated.append(x1)
        ry_interpolated.append(y1)
        rx_interpolated.append(xm)
        ry_interpolated.append(ym)
    # 마지막 포인트 추가
    # 마지막 포인트가 보간 과정에서 누락되는 것을 방지한다.
    rx_interpolated.append(rx[-1])
    ry_interpolated.append(ry[-1])
    return rx_interpolated, ry_interpolated

def planning(sx, sy, syaw, max_acceleration, dt):
    global rx, ry
    global MAP_X1, MAP_Y1
    MAP_X1, MAP_Y1 = pygame.display.get_surface().get_size() #맵의 끝 좌표 초기화
    print("Start Planning")
    xycar_radian = syaw_to_radian(syaw)
    print("sx : ", sx, "sy : ", sy, "xycar_radian : ", xycar_radian," Entry_Point : ", P_ENTRY)
    #시작 node와 목적 node 전달, obstacle_point를 초기화하기 위한 시작 기울기와 목적 기울기를 전달
    path = aStar((sx, sy) , P_ENTRY, -1* math.tan(xycar_radian), -1)
    rx = [p[0] for p in path]
    ry = [p[1] for p in path]
    rx, ry = interpolate_path(rx, ry)

    end_path = aStar(P_ENTRY, P_END, -1, -1)
    end_x = [p[0] for p in end_path]
    end_y = [p[1] for p in end_path]
    rx += end_x
    ry += end_y
    return rx, ry


#=============================================
# 생성된 경로를 따라가는 함수
# 파이게임 screen, 현재위치 x,y 현재각도, yaw
# 현재속도 velocity, 최대가속도 max_acceleration 단위시간 dt 를 전달받고
# 각도와 속도를 결정하여 주행한다.
#=============================================

### tracking ###
"""
### PID 제어 알고리즘 사용을 사용한다
1. 차량의 현재 위치에서 경로 상의 점들 중 가장 가까운 좌표을 찾고, (find_nearest_point_index)
2. 목표 좌표와 각도 차이를 PID 제어를 통해 계산하여 조향각을 결정한다. (calculate_angle)
3. 현재 위치와 목표 위치 간의 거리 차이를 기반으로 전진/후진을 결정한다. (drive_direction)
"""

def find_nearest_point_index(x, y, path):
# 차량의 현재 위치와 주어진 경로 상의 포인트들 사이에서 가장 가까운 포인트의 인덱스를 찾아내는 함수
# ----------------------------------------------------
# 1. 현재 위치와 모든 포인트들 사이의 거리를 계산한다
# 2. 그 중 가장 가까운 포인트의 인덱스를 찾는다
# ----------------------------------------------------
    # 현재 차량 위치(x, y)와 경로(path)의 모든 포인트들 사이의 거리를 계산한다.
    dx = [x - px for px in path[0]] # x좌표와 모든 경로의 x좌표들 사이의 차이(dx)
    dy = [y - py for py in path[1]] # y좌표와 모든 경로의 y좌표들 사이의 차이(dy)
    # 계산된 거리들 중에서 가장 가까운 점의 인덱스를 찾는다
    distances = np.hypot(dx, dy)
    nearest_point_index = np.argmin(distances)
    return nearest_point_index

class PID:
# PID 제어 구현
# 차량의 조향각을 조절하기 위해 사용된다.
# 현재 각도와 목표 각도 사이의 차이를 기반으로 조향각을 계산한다
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp # 비례 제어
        self.Ki = Ki # 적분 제어
        self.Kd = Kd # 미분 제어
        self.prev_error = 0 # 이전 오차값
        self.integral = 0 # 누적 오차값

    def control(self, error):
        # 누적 오차값 현재 오차값을 업데이트한다
        self.integral += error
        # 현재 오차값과 이전 오차의 차이를 통해 오차의 변화율을 계산한다
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.prev_error = 0
        self.integral = 0

# PID 제어에 사용할 값을 설정한다.
pid = PID(Kp=200, Ki=0.02, Kd=20)

def calculate_angle(x, y, yaw, path, Lf):
# 차량이 경로를 따라가기 위해 필요한 조향각을 계산하는 함수
# ----------------------------------------------------
# 1. 현재 위치에서 가장 가까운 경로상의 포인트을 찾고 (find_nearest_point_index)
# 2. 차량의 이동방향을 고려하여 목표 포인트와의 각도 차이를 계산한다
# 3. 각도차이를 PID 제어를 통해 최종적으로 차량의 조향각을 결정한다 (PID)
# ----------------------------------------------------
    global rx, ry
    # 현재위치에서 경로 상의 가장 가까운 포인트을 찾고 목표 포인트을 설정한다.
    nearest_point_index = find_nearest_point_index(x, y, (rx, ry))
    # 가장 가까운 포인트의 인덱스부터 경로 업데이트
    # 지나온 경로는 삭제하고 남은 경로만 고려한다.
    rx=rx[nearest_point_index:]
    ry=ry[nearest_point_index:]
    # Lf와 경로 길이 중 더 작은 값을 target_index로 설정한다.
    # 배열 범위를 벗어나지 않도록 한다
    target_index = min(Lf, len(rx) - 1)
    # 목표 포인트와 현재 위치 사이의 각도 계산
    target_x = rx[target_index]
    target_y = ry[target_index]
    yaw = np.deg2rad(-yaw)
    # 목표 지점 사이의 각도 차이를 계산하고
    # PID 제어를 통해 조향각을 설정한다
    alpha = math.atan2(target_y - y, target_x - x) - yaw
    delta = pid.control(alpha)
    # 조향각을 +/- 20도 제한한다
    if delta > 20:
        delta = 20
    elif delta < -20:
        delta = -20
    return delta

def drive_direction(x, y, yaw, velocity, dt, Lf):
# 목표 경로 상의 위치에 따라 차량이 전진해야 할지 후진해야 할지 결정하는 함수
# ----------------------------------------------------
# 1. dt동안 이동할 수 있는 거리를 계산한다.
# 2. 차량이 1) 전진했을때 예상위치, 2)후진했을때 예상위치를 계산한다.
# 3. 차량의 다음경로 포인트와 전진 예상위치, 후진 예상위치의 차이를 각각 계산한다
# 4. 차량이 후진했을 때 목표위치에 더 가까워진다면 후진으로 설정한다.
# ----------------------------------------------------
    global rx, ry
    target_idx = min(Lf, len(rx) - 1)
    rad = np.deg2rad(-yaw)
    # 현재 위치에서 Lf만큼 떨어진 곳의 거리를 계산한다
    x_step = math.cos(rad)*velocity*dt
    y_step = math.sin(rad)*velocity*dt
    forward_point = (x+Lf*x_step, y+Lf*y_step) # 차량이 전진했을때, 예상위치
    backward_point = (x-Lf*x_step, y-Lf*y_step) # 차량이 후진했을때, 예상위치
    # 목표위치와 예상위치의 거리차이를 계산한다
    forward_distance = distance((rx[target_idx], ry[target_idx]), forward_point)
    backward_distance = distance((rx[target_idx], ry[target_idx]), backward_point)
    # 만약 목표 전진 거리가 목표 후진 거리보다 작거나 같으면 기존의 주행 방향을 유지한다.
    if forward_distance <= backward_distance:
        return velocity
    # 차이가 유의미하지 않다면 기존의 주행 방향을 유지한다.
    if abs(forward_distance-backward_distance) < Lf:
        return velocity
    # 속도의 부호를 반전시킨다.
    return -1*velocity

def tracking(screen, x, y, yaw, velocity, max_acceleration, dt):
# calculate_angle 함수와 drive_direction 함수를 호출하여 차량의 주행을 제어한다
# ----------------------------------------------------
# 1. 차량의 현재 전진, 후진 상태를 기준으로 차량의 현재위치를 조정한다.
# 2. 조향각을 계산한다.
# 3. 차량의 이동 방향을 결정한다.
# 4. 주차공간에 도착하면 차량을 정지시킨다.
# ----------------------------------------------------
    global rx, ry
    Lf = 20  # 추종거리
    # 차량이 들어가는 P_ENTRY와 P_END의 거리 차이는 약 131이므로
    # 차량의 길이 절반을 (131 // 2) = 65라고 가정한다
    if velocity == 0 :
        velocity = 50
    # 전진하고 있을 경우, 현재위치를 차량의 길이 절반(65)만큼 앞으로 조정한다
    # -> 차량의 앞부분이 경로 포인트를 따라가도록 제어한다.
    if velocity > 0:
        #pygame에서는 y축 방향이 반대이기 때문에 yaw에 -1을 곱하여 처리한다.
        x += math.cos(np.deg2rad(-yaw)) * 65
        y += math.sin(np.deg2rad(-yaw)) * 65
    # 후진하고 있을 경우, 현재위치를 차량의 길이 절반(65)만큼 뒤로 조정한다
    # -> 차량의 뒷부분이 경로 포인트를 따라가도록 제어한다.
    else :
        x -= math.cos(np.deg2rad(-yaw)) * 65
        y -= math.sin(np.deg2rad(-yaw)) * 65
    # calculate_angle 함수로부터 조향각을 계산한다.
    angle = calculate_angle(x, y, yaw, (rx, ry), Lf)
    # drive_direction 함수로부터 차량의 방향을 결정한다.
    speed = drive_direction(x,y,yaw,velocity,dt, Lf)
    # 만약 경로의 길이가 1이하인 경우 속도를 0으로 설정하여 차량을 정지시킨다.
    if len(rx) <= 1:
        speed = 0
    drive(angle, speed)
