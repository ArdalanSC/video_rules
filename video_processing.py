from object_detection import YoloModel
from PIL import Image
from tracker import CentroidTracker, ShotDetection
from datetime import datetime
import json
import cv2
import os 


class VideoProcessor:
    def __init__(self, file_path, save_dir="results"):

        self.file_path = file_path
        file_name = os.path.basename(file_path).split(".")[0]
        now = datetime.now().strftime("%Y%m%d__%H%M%S")
        self.folder_name = os.path.join(save_dir, f"{now}_{file_name}")
        os.makedirs(self.folder_name)

        self.vid_stream = cv2.VideoCapture(file_path)
        self.frame_width = int(self.vid_stream.get(3))
        self.frame_height = int(self.vid_stream.get(4))
        self.fps = self.vid_stream.get(cv2.CAP_PROP_FPS)
        self.total_num_frames = self.vid_stream.get(cv2.CAP_PROP_FRAME_COUNT)

        self.model = YoloModel("yolov5l.yaml", "yolov5l.pt")
        self.tracker = CentroidTracker()
        self.shot_detector = ShotDetection(self.folder_name)

        vid_info = {"file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "total_frames": self.total_num_frames,
                    "fps": self.fps,
                    "length": self.total_num_frames/self.fps,
                    "width": self.frame_width,
                    "height": self.frame_height}

        vid_info_path = os.path.join(self.folder_name, "vid_info.json")
        with open(vid_info_path, "w+") as f:
            json.dump(vid_info, f)

    def run(self):
        shot_num = 0
        shot_folder = os.path.join(self.folder_name, f"shot_{shot_num}")
        os.makedirs(shot_folder)

        while True:
              
            ret, frame = self.vid_stream.read()

            if not ret:
                break

            frame_num = int(self.vid_stream.get(cv2.CAP_PROP_POS_FRAMES))

            if self.shot_detector(frame, frame_num, frame_num/self.fps):
                self.tracker.reset()
                shot_num += 1
                shot_folder = os.path.join(self.folder_name, f"shot_{shot_num}")
                os.makedirs(shot_folder)

            pil_img = Image.fromarray(frame.astype('uint8'), 'RGB')

            result = self.model(pil_img)

            self.tracker.update(result, frame, frame_num, shot_folder)

        self.vid_stream.release()
        self.shot_detector.release()
        self.tracker.release()

        return self.folder_name


VideoProcessor("cut_1.mp4").run()
