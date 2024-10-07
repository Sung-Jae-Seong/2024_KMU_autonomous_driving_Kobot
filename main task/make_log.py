import time
import json
import sys
import signal
from datetime import datetime
import os
import cv2

class Logging:
    def __init__(self, filename='/home/xytron/log.json'):
        self.prev_time = time.time()
        self.filename = filename
        self.img_path = "/home/xytron/xycar_imgs/img"
        self.processed_path = "/home/xytron/xycar_imgs/processed"
        self.log_entries = []  # 메모리에 로그 데이터를 저장할 리스트
        self.log_entries.append({
                                "image" : None,
                                "time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "position": (0,0),
                                "direction": 0,
                                "speed": 0,
                                "lidar": [],
                                "AR_Tag": {}
                                })
        self.delete_imgs(self.img_path[:-4])

        # SIGINT 시그널 처리 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)

    def delete_imgs(self, path):
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path,  file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print("Error : ", e)

    def write_log(self, log_entry):
        if time.time() -  self.prev_time < 0.03:
            return
        self.prev_time = time.time()
        img_filename = self.img_path + str(len(self.log_entries)) + ".jpg"
        processed_filename = self.processed_path + str(len(self.log_entries)) + ".jpg"

        if log_entry["image"] is not None:
            cv2.imwrite(img_filename, log_entry["image"])
            log_entry["image"] = img_filename
        if log_entry["processed"] is not None:
            cv2.imwrite(processed_filename, log_entry["processed"])
            log_entry["processed"] = processed_filename

        self.log_entries.append(log_entry)

    def save_logs_to_file(self):
        # 프로그램 종료 시 모든 로그를 파일에 저장
        with open(self.filename, 'w', encoding='utf-8') as file:
            json.dump(self.log_entries, file, indent=4)
        print("#### log is saved")

    def signal_handler(self, sig, frame):
        print('Ctrl+C was pressed. Exiting gracefully...')
        self.save_logs_to_file()
        sys.exit(0)
