import numpy as np
import cv2

class TrafficSignDetector:
    def __init__(self, min_R=15, max_R=25, center_x=340, center_y=100, half_W=300, half_H=80):
        self.min_radius = min_R
        self.max_radius = max_R
        self.center_x = center_x
        self.center_y = center_y
        self.half_width = half_W
        self.half_height = half_H

    def detect_circles(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=40, param2=20,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        return circles

    def analyze_brightness(self, circles, gray_img):
        max_mean_value = 0
        brightest_circle = None
        for (x, y, r) in circles:
            roi = gray_img[y - (r // 2):y + (r // 2), x - (r // 2):x + (r // 2)]
            mean_value = np.mean(roi)
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                brightest_circle = (x, y, r)
        return brightest_circle, max_mean_value

    def check_circle_distribution(self, circles):
        y_sorted_circles = sorted(circles, key=lambda c: c[1])
        x_sorted_circles = sorted(circles, key=lambda c: c[0])

        vertical_diff = self.max_radius * 2
        if len(y_sorted_circles) > 1 and (y_sorted_circles[-1][1] - y_sorted_circles[0][1]) > vertical_diff:
            return False, "Circles are scattered vertically"

        horizontal_diff = self.max_radius * 8
        if (x_sorted_circles[-1][0] - x_sorted_circles[0][0]) > horizontal_diff:
            return False, "Circles are scattered horizontally"

        min_distance = self.min_radius * 3
        for i in range(len(x_sorted_circles) - 1):
            if (x_sorted_circles[i + 1][0] - x_sorted_circles[i][0]) < min_distance:
                return False, "Circles are too close horizontally"

        return True, None

    def check_traffic_sign(self, image):
        roi_img = image[self.center_y - self.half_height:self.center_y + self.half_height,
                        self.center_x - self.half_width:self.center_x + self.half_width]

        circles = self.detect_circles(roi_img)

        if circles is None or len(circles[0]) != 3:
            return False

        circles = np.round(circles[0, :]).astype("int")
        brightest_circle, max_mean_value = self.analyze_brightness(circles, cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))

        if brightest_circle is not None:
            is_valid, error_message = self.check_circle_distribution(circles)
            if not is_valid:
                return False

            if np.argmax([x[0] for x in circles]) == 2:  # Example check
                return True  # Traffic Sign is Blue

        return False  # Traffic Sign is NOT Blue
