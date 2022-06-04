from scipy.spatial import distance as dist
from collections import OrderedDict
from PIL import Image
import numpy as np
import json
import cv2
import os


class CentroidTracker:
    def __init__(self, maxDisappeared=15):
        self.nextObjectID = 0
        self.objects = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect, _class, frame, frame_num, directory):
        new_obj = TrackingObject(self.nextObjectID, centroid, rect, _class, frame, frame_num, directory)
        self.objects[self.nextObjectID] = new_obj
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects[objectID].deregister()
        del self.objects[objectID]

    def update(self, detections, frame, frame_num, directory):
        rects = []
        classes = []
        for item in detections:
            rects.append(item["coordinate"])
            classes.append(item["label"])
        return self.__update(rects, classes, frame, frame_num, directory)

    def __update(self, rects, classes, frame, frame_num, directory):
        if len(rects) == 0:
            for objectID, obj in self.objects.items():
                obj.disappeared()
                if obj.num_disappeared > self.maxDisappeared:
                    self.deregister(objectID)
 
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
 
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i], classes[i], frame, frame_num, directory)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(obj.center for obj in self.objects.values())
             
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
 
            rows = D.min(axis=1).argsort()
 
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
 
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
 
                objectID = objectIDs[row]
                self.objects[objectID].add(inputCentroids[col], rects[col], classes[col], frame, frame_num)
 
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.objects[objectID].disappeared()
 
                    if self.objects[objectID].num_disappeared > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col], classes[col], frame, frame_num, directory)

        return self.objects

    def reset(self):
        for objectID in list(self.objects.keys()):
            self.deregister(objectID)

    def release(self):
        return self.reset()


class TrackingObject:
    def __init__(self, _id, center, rect, _class, frame, frame_num, directory):
        self.id = _id
        self.centers = [center.tolist()]
        self.rects = [rect]
        self.classes = [_class]
        self.cropped_objs = [self._crop(frame, rect)]
        self.frame_nums = [frame_num]
        self.directory = directory
        self.num_disappeared = 0

    def add(self, center, rect, _class, frame, frame_num):

        self.centers.append(center.tolist())
        self.rects.append(rect)
        self.classes.append(_class)
        self.cropped_objs.append(self._crop(frame, rect))
        self.frame_nums.append(frame_num)

        self.num_disappeared = 0

    @property
    def center(self):
        return self.centers[-1]

    def disappeared(self):
        self.num_disappeared += 1

    def deregister(self):
        data = {"object_id": self.id,
                "frames" : []}

        object_path = os.path.join(self.directory, f"object_{self.id}")
        os.makedirs(object_path)

        prev_center = None
        for center, rect, _class, img, frame_num in zip(self.centers,
                                                        self.rects,
                                                        self.classes,
                                                        self.cropped_objs,
                                                        self.frame_nums):

            img_name = os.path.join(self.directory, f"{self.id}_{frame_num}.jpg")
            if prev_center is None:
                distance_to_prev_center = 0
                rel_dist_to_prev_center = 0
            else:
                distance_to_prev_center = self._dist(center, prev_center)
                rel_dist_to_prev_center = center[0] - prev_center[0]

            d = {"frame_num": frame_num,
                 "center": center,
                 "rect": rect,
                 "class": _class,
                 "distance_to_prev_frame": distance_to_prev_center,
                 "rel_dist_to_prev_frame": rel_dist_to_prev_center}
            data["frames"].append(d)
            file_name = os.path.join(object_path, f"O{self.id}_F{int(frame_num)}.jpg")
            cv2.imwrite(file_name, img)
        jason_file_name = os.path.join(object_path, f"O{self.id}.json")
        with open(jason_file_name, "w+") as f:
            json.dump(data, f)


    def get_direction(self, n):
        # Get direction (left/right) of object of the last n frames

        threshold = 20

        length = len(self.centers)

        if length == 1:
            return None

        if n > length:
            n = length

        first_center = self.centers[-n]
        last_center = self.centers[-1]

        if (first_center[0] - last_center[0]) > threshold:
            return "Left"
        elif first_center[0] - last_center[0] < - threshold:
            return "Right"
        else:
            return "Stationary"

    @staticmethod
    def _crop(frame, rect):
        x1, y1, x2, y2 = rect
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _dist(c1, c2):
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5