import numpy as np

class ObstacleDrive:
    def __init__(self):
        self.lidar_points = []
        self.safe_distance = 0.001

    def valid_mean(self, points):
        points = np.array(points)
        points[points == 0] = 100
        points[np.isinf(points)] = 100
        valid_points = points[np.isfinite(points) & (points <= 0.6)]
        return np.mean(valid_points) if len(valid_points) > 0 else 100

    def obstacle_lane(self, lidar_points):
        left_distance = self.valid_mean(lidar_points[30:90])
        right_distance = self.valid_mean(lidar_points[630:690])
        distances = {'left': left_distance, 'right': right_distance}

        if abs(left_distance - right_distance) < self.safe_distance:
            return 'middle'
        # 평균값 중 가장 큰 쪽으로
        obstacle_lane = min((k for k, v in distances.items() if v != 0.0), key=distances.get)
        return obstacle_lane
