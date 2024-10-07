import math
import time
import cv2
import numpy as np

class AR:
    def __init__(self):
        self.prev_data = {"ID": None, "DX": None, "DZ": None}
        self.K_ar = 120 * (-1) # 커질수록 더 급격하게 회전
        self.ar_time = time.time()
        self.ar_flag = 0
        self.prev_id = None
        self.prev_distance = None
        self.image = None

    def reset(self):
        self.prev_data = {"ID": None, "DX": None, "DZ": None}
        self.ar_flag = 0
        self.ar_time = time.time()
        self.prev_id = None

    def set_img(self, image):
        self.image = image

    def check_ar_flag(self, ID, distance):
        if (time.time() - self.ar_time > 1.5
            and self.prev_id is not None
            and (
                ID != self.prev_id
                or abs(distance - self.prev_distance) > 70)
            ):
            self.ar_flag += 1
            self.ar_time = time.time()

    def is_mask(self):
        h, w = self.image.shape[:2]
        copy_image = self.image[50:h - 50, 50:w-50]
        hsv = cv2.cvtColor(copy_image, cv2.COLOR_BGR2HSV)
        v_std = np.std(hsv[:, :, 2])
        return v_std < 35

    def filter_ar_data(self, ar_data):
        closest_data = {"ID": None, "DX": None, "DZ": None}
        min_distance = float('inf')

        for i in range(len(ar_data["ID"])):
            ar_id = ar_data["ID"][i]
            dx = ar_data["DX"][i]
            dz = ar_data["DZ"][i]
            # 이전 dx와 dz 값 가져오기
            prev_dx, _ = self.prev_data.get(ar_id, (None, None))
            # 55값보다 클 때만 기준으로 삼음
            if prev_dx is not None and abs(dx - prev_dx) >= 55:
                dx = prev_dx

            distance = math.sqrt(dx**2 + dz**2)
            #가장 가까운 것을 선택
            if distance < min_distance and dz > 25:
                closest_data["ID"] = ar_id
                closest_data["DX"] = dx
                closest_data["DZ"] = dz
                min_distance = distance

        if closest_data["ID"] is not None:
            self.prev_data[closest_data["ID"]] = (closest_data["DX"], closest_data["DZ"])
        elif len(ar_data["ID"]) > 0:
            last_id = ar_data["ID"][-1]
            closest_data = {
                "ID": last_id,
                "DX": self.prev_data.get(last_id, (None, None))[0],
                "DZ": self.prev_data.get(last_id, (None, None))[1]
            }

        return closest_data

    def ar_drive(self, filtered_data):
        if filtered_data["ID"] is None:
            return "finish"

        dx = filtered_data["DX"]
        dz = filtered_data["DZ"]

        target_x = 45 if dx < 0 else -45
        target_z = 10

        distance = math.sqrt((dx - target_x) ** 2 + (dz - target_z) ** 2)
        self.prev_id = filtered_data["ID"]
        self.prev_distance = distance

        if self.ar_flag == 1:
            if distance > 91:
                error = dx - 30
            else:
                error = abs((dx - target_x) / distance) * self.K_ar
        elif self.ar_flag == 2:
            if distance > 69:
                error = dx - 27
            else:
                error = abs((dx - target_x) / distance) * self.K_ar
        elif self.ar_flag == 3:
            if distance > 60:
                error = abs(dx + 2) * (-1)
            else:
                error = abs((dx - target_x) / distance) * self.K_ar
        elif self.ar_flag == 4:
            if distance > 90:
                error = dx - 27
            else:
                error = abs((dx - target_x) / distance) * self.K_ar
        else:
            if distance > 76:
                error = dx - 27
            else:
                error = abs((dx - target_x) / distance) * self.K_ar
        return error

    def traffic_ar_drive(self, filtered_data):
        if filtered_data["ID"] is None:
            if self.is_mask():
                return "is_mask"
            return -8

        dx = filtered_data["DX"]
        dz = filtered_data["DZ"]

        target_x = 50 if dx < 0 else -50
        target_z = 10

        distance = math.sqrt((dx - target_x) ** 2 + (dz - target_z) ** 2)

        if distance > 164:
            return dx
        return "mode_change"
