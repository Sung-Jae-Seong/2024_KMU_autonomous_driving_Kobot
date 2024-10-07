import numpy as np
import cv2

import slide_window
import camera_processing

class LaneDetector:
    def __init__(self):
        self.camera_processor = camera_processing.CameraProcessing()
        self.slide_window_processor = slide_window.SlideWindow(None)
        self.processed_frame = None
        self.left_history = []
        self.right_history = []
        self.max_length = 3  # 중위수 필터의 길이

    def reset(self):
        self.slide_window_processor.reset()

    def median_filter(self, values):
        return int(np.median(values))

    def detect(self, frame):
        frame, filtered= self.camera_processor.process_image(frame)

        if frame is not None: #processed를 processed_frame으로 바꾸기
            detected, left, right, processed = self.slide_window_processor.slide(frame, filtered)
            if len(self.left_history) >= self.max_length:
                self.left_history.pop(0)
            if len(self.right_history) >= self.max_length:
                self.right_history.pop(0)

            self.left_history.append(left)
            self.right_history.append(right)

            left = self.median_filter(self.left_history)
            right = self.median_filter(self.right_history)

            cv2.circle(processed, (left, frame.shape[0] // 2), 4, (0, 0, 255), -1)
            cv2.circle(processed, (right, frame.shape[0] // 2), 4, (0, 255, 0), -1)
            self.processed_frame = processed
            return detected, left, right
        return False, None, None, frame
