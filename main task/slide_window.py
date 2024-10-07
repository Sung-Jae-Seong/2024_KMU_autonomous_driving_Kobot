import numpy as np
import cv2


class SlideWindow:
    def __init__(self, control_node=None):
        self.control_node = control_node  # 제어 노드 객체
        self.center_old = None  # 이전 중앙값
        self.left_old = None  # 이전 왼쪽 값
        self.right_old = None  # 이전 오른쪽 값
        self.sobel_filter = ""
        self.false_counter = 0

    def reset(self):
        self.center_old = None
        self.left_old = None
        self.right_old = None

    def preprocess(self, img):
        if self.center_old is None or self.false_counter > 10:
            self.center_old = 320
            self.left_old = 140
            self.right_old = 500
            self.false_counter = 0
            return img

        height, width = img.shape[:2]
        threshold = 90

        left_min = left_max = right_min = right_max = 0

        if self.sobel_filter == 'vertical':
            left_min = max(self.left_old - threshold//2, 0)
            left_max = min(self.left_old + threshold//2, width)
            right_min = max(self.right_old - threshold//2, 0)
            right_max = min(self.right_old + threshold//2, width)
        elif self.sobel_filter in ('diagonal_1', 'diagonal_3'):
            left_min = max(self.left_old - threshold, 0)
            left_max = min(self.left_old + threshold//3, width)
            right_min = max(self.right_old - threshold, 0)
            right_max = min(self.right_old + threshold//3, width)
        elif self.sobel_filter in ('diagonal_2', 'diagonal_4'):
            left_min = max(self.left_old - threshold//3, 0)
            left_max = min(self.left_old + threshold, width)
            right_min = max(self.right_old - threshold//3, 0)
            right_max = min(self.right_old + threshold, width)

        mask = np.zeros_like(img)

        mask[:, left_min:left_max] = img[:, left_min:left_max]
        mask[:, right_min:right_max] = img[:, right_min:right_max]

        return mask

    def slide(self, img, filtered):
        # 주어진 이미지에서 슬라이딩 윈도우를 사용하여 차선을 감지하는 함수
        self.sobel_filter = filtered
        img = self.preprocess(img)
        #flag = 0
        #height, width = img.shape

        roi_img = img.copy()  # 관심영역 설정
        roi_height, roi_width = roi_img.shape
        c_img = np.dstack((roi_img, roi_img, roi_img))  # 컬러 이미지 생성

        window_height = 15  # 윈도우 높이
        window_width = 30  # 윈도우 너비
        minpix = 40 # 최소 라인 길이
        n_windows = roi_width // 2  # 윈도우 개수
        pts_center = np.array([[roi_width // 2, 0], [roi_width // 2, roi_height]], np.int32)
        cv2.polylines(c_img, [pts_center], False, (0, 120, 120), 1) # 중앙선 그리기

        nonzero = roi_img.nonzero()  # 관심영역의 비제로 좌표 찾기
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_center = self.center_old
        y_center = roi_height // 2

        left_idx = 0
        right_idx = 0

        find_left = False
        find_right = False

        left_start_x = 0
        #left_start_y = 0

        right_start_x = 0
        #right_start_y = 0

        dist_threshold = 340
        dist = None

        win_L_y_low = 0
        win_L_y_high = 0
        win_L_x_low = 0
        win_L_x_high = 0
        win_R_y_low = 0
        win_R_y_high = 0
        win_R_x_low = 0
        win_R_x_high = 0

        for i in range(0, n_windows):
            if find_left is False:
                win_L_y_low = y_center - window_height // 2
                win_L_y_high = y_center + window_height // 2

                win_L_x_high = x_center - left_idx * window_width
                win_L_x_low = x_center - (left_idx + 1) * window_width

            if find_right is False:
                win_R_y_low = y_center - window_height // 2
                win_R_y_high = y_center + window_height // 2

                win_R_x_low = x_center + right_idx * window_width
                win_R_x_high = x_center + (right_idx + 1) * window_width

            cv2.rectangle(c_img, (win_L_x_low, win_L_y_low), (win_L_x_high, win_L_y_high), (0, 255, 0), 1)
            cv2.rectangle(c_img, (win_R_x_low, win_R_y_low), (win_R_x_high, win_R_y_high), (0, 0, 255), 1)

            good_left_inds = (
                (nonzeroy >= win_L_y_low) & (nonzeroy < win_L_y_high) &
                (nonzerox >= win_L_x_low) & (nonzerox < win_L_x_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_R_y_low) & (nonzeroy < win_R_y_high) &
                (nonzerox >= win_R_x_low) & (nonzerox < win_R_x_high)
            ).nonzero()[0]

            if len(good_left_inds) > minpix and find_left is False:
                find_left = True
                left_start_x = np.int32(np.mean(nonzerox[good_left_inds]))
                left_start_y = roi_height // 2
                for i, e in enumerate(good_left_inds):
                    cv2.circle(c_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0, 255, 0), -1)
            else:
                left_idx += 1

            if len(good_right_inds) > minpix and find_right is False:
                find_right = True
                right_start_x = np.int32(np.mean(nonzerox[good_right_inds]))
                right_start_y = roi_height // 2
                for i, e in enumerate(good_right_inds):
                    cv2.circle(c_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (0, 0, 255), -1)
            else:
                right_idx += 1

            # 1. 차선이 왼쪽, 오른쪽 둘 다 인식된 경우
            if find_left and find_right:
                dist = right_start_x - left_start_x
                self.center_old = np.int32((right_start_x + left_start_x) / 2)
                if dist_threshold < dist < dist_threshold + 80:
                    self.left_old = left_start_x
                    self.right_old = right_start_x

                dist_from_old_L = abs(left_start_x - self.left_old)
                dist_from_old_R = abs(right_start_x - self.right_old)

                if dist_from_old_L < 30 and dist_from_old_R < 30:
                    self.center_old = np.int32((left_start_x + right_start_x) / 2)

                    # 커브 감지: 차선 간 거리와 각도 변화 모두 확인
                    if abs(left_start_x - right_start_x) < dist_threshold - 20:
                        if self.control_node:
                            self.control_node.handle_curve_detected()  # CarlaControlNode의 메서드를 호출
                    else:
                        if self.control_node:
                            self.control_node.handle_curve_notdetected()

                elif dist_from_old_L < 30:
                    find_right = False
                    right_idx += 1
                    self.center_old = left_start_x + dist_threshold // 2
                    right_start_x = left_start_x + dist_threshold

                elif dist_from_old_R < 30:
                    find_left = False
                    left_idx += 1
                    self.center_old = right_start_x - dist_threshold // 2
                    left_start_x = right_start_x - dist_threshold
                return 'both', left_start_x, right_start_x, c_img

            # 2. 차선이 왼쪽만 인식된 경우
            if find_left:
                right_idx += 1
                dist_from_center = self.center_old - left_start_x
                dist_from_old = abs(left_start_x - self.left_old)

                if dist_from_center > 50 and dist_from_old < 70:
                    self.center_old = left_start_x + (dist_threshold//2)
                    change = left_start_x - self.left_old
                    right_start_x = self.right_old + change
                    self.left_old = left_start_x
                    self.right_old = right_start_x
                    return 'left', left_start_x, right_start_x, c_img

            # 3. 차선이 오른쪽만 인식된 경우
            elif find_right:
                find_left = False
                left_idx += 1
                dist_from_center = right_start_x - self.center_old
                dist_from_old = abs(right_start_x - self.right_old)

                if dist_from_center > 50 and dist_from_old < 70:
                    self.center_old = right_start_x - (dist_threshold//2)
                    change = right_start_x - self.right_old
                    left_start_x = self.left_old + change
                    self.left_old = left_start_x
                    self.right_old = right_start_x
                    return 'right', left_start_x, right_start_x, c_img

        # 4. 차선이 둘 다 인식되지 않은 경우
        self.false_counter += 1
        return False, self.left_old, self.right_old, c_img
