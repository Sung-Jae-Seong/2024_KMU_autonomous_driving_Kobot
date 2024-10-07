import numpy as np

class SensorDrive:
    def __init__(self):
        self.safe_distance = 0.176
        self.k = 30.0

    def valid_min(self, points):
        points = np.array(points)
        valid_points = points[np.isfinite(points)]
        if len(valid_points) > 0:
            return np.min(valid_points)
        return np.inf

    def sensor_error(self, lidar_points):
        #===================라바콘 변수 ==================
        # 장애물 판단 범위
        left_points = lidar_points[90:160] # 0~180
        right_points = lidar_points[560:610] # 540~719
        # 평균값으로 변환
        left_distance = self.valid_min(left_points)
        right_distance = self.valid_min(right_points)
        error = 0

        if np.inf not in (left_distance, right_distance): # 둘 다 inf가 아닐 때
            # 왼쪽 거리 너무 가까울 경우 오른쪽으로 회전
            if left_distance < 0.22 or right_distance == 0.0:
                error = 30
            # 오른쪽 거리 너무 가까울 경우 왼쪽으로 회전
            elif right_distance < 0.22 or left_distance == 0.0:
                error = -30
            # 왼쪽으로 가야할 때
            elif left_distance - self.safe_distance > right_distance:
                error -= 4.0  - right_distance
                if error > 0:
                    error = error*(-1)
            # 오른쪽으로 가야할 때
            elif right_distance - self.safe_distance > left_distance:
                error += 4.0 - left_distance
                if error < 0:
                    error = error*(-1)
            else:
                error = 0
        else: # left_distance 또는 right_distance가 inf인 경우
            if left_distance == np.inf:
                error -= 4.0 - right_distance  # 왼쪽으로 회전
            elif right_distance == np.inf:
                error += 4.0  - left_distance   # 오른쪽으로 회전
        return error * self.k
