from PIL import Image
from . import imagehash
import json
import os


class ShotDetection:
    def __init__(self, directory="", threshold=350):
        self.threshold = threshold
        self._prev_hash = None
        self.history = []
        self.directory = directory

    def __call__(self, frame, frame_num, frame_sec):
        frame_hash = imagehash.average_hash(
            Image.fromarray(frame[::-1]), hash_size=32)
        if self._prev_hash is None:
            self._prev_hash = frame_hash
            return False
        diff = self._prev_hash - frame_hash
        self._prev_hash = frame_hash
        self.history.append((frame_num, frame_sec, diff))
        return diff > self.threshold

    def release(self):
        file_name = os.path.join(self.directory, "frame_diffs.json")
        with open(file_name, "w+") as f:
            json.dump(self.history, f)