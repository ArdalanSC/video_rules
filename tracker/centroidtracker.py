from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 

class CentroidTracker:
    def __init__(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect, _class):
        new_obj = TrackingObject(self.nextObjectID, centroid, rect, _class)
        self.objects[self.nextObjectID] = new_obj
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects[objectID].deregister()
        del self.objects[objectID]

    def update(self, detections):
        rects = []
        classes = []
        for item in detections:
            rects.append(item["coordinate"])
            classes.append(item["label"])
        return self.__update(rects, classes)

    def __update(self, rects, classes):
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
                self.register(inputCentroids[i], rects[i], classes[i])
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
                self.objects[objectID].add(inputCentroids[col], rects[col], classes[col])
 
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
                    self.register(inputCentroids[col], rects[col], classes[col])

        return self.objects


class TrackingObject:
    def __init__(self, id, center, rect, _class):
        self.id = id
        self.centers = [center]
        self.rects = [rect]
        self.classes = [_class]
        self.num_disappeared = 0

    def add(self, center, rect, _class):
        self.centers.append(center)
        self.rects.append(rect)
        self.classes.append(_class)
        self.num_disappeared = 0

    @property
    def center(self):
        return self.centers[-1]

    def disappeared(self):
        self.num_disappeared += 1

    def deregister(self):
        pass

    def get_direction(self, n):
        # Get direction (left/right) of object of the last n frames

        length = len(self.centers)

        if length == 1:
            return None

        if n > length:
            n = length

        first_center = self.centers[-n]
        last_center = self.centers[-1]

        if (first_center[0] - last_center[0]) > 0:
            return "Left"
        elif first_center[0] == last_center[0]:
            return "Stationary"
        else:
            return "Right"