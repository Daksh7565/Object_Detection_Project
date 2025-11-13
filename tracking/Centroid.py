# --- NEW: Imports for the Centroid Tracker ---
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import numpy as np


# =====================================================================================
# --- NEW: CENTROID TRACKER CLASS ---
# =====================================================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=20, history_size=10, confirm_hits=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.class_history = OrderedDict()
        self.confidence_history = OrderedDict()
        self.radius_history = OrderedDict()
        self.hits = OrderedDict()
        self.status = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.history_size = history_size
        self.confirm_hits = confirm_hits

    def register(self, detection):
        centroid, radius, class_id, confidence = detection
        objectID = self.nextObjectID
        self.objects[objectID] = centroid
        self.disappeared[objectID] = 0
        self.class_history[objectID] = deque([class_id], maxlen=self.history_size)
        self.confidence_history[objectID] = deque([confidence], maxlen=self.history_size)
        self.radius_history[objectID] = deque([radius], maxlen=self.history_size)
        self.hits[objectID] = 1
        self.status[objectID] = 'tentative'
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.class_history[objectID]
        del self.confidence_history[objectID]
        del self.radius_history[objectID]
        del self.hits[objectID]
        del self.status[objectID]

    def get_confirmed_objects(self):
        confirmed_objects = OrderedDict()
        for objectID, current_status in self.status.items():
            if current_status == 'confirmed':
                voted_class_id = max(set(self.class_history[objectID]), key=list(self.class_history[objectID]).count)
                avg_confidence = np.mean(self.confidence_history[objectID])
                avg_radius = int(np.mean(self.radius_history[objectID]))
                confirmed_objects[objectID] = {
                    'centroid': self.objects[objectID],
                    'radius': avg_radius,
                    'class_id': voted_class_id,
                    'confidence': avg_confidence
                }
        return confirmed_objects

    def update(self, detections):
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                self.hits[objectID] = 0
                self.status[objectID] = 'tentative'
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.get_confirmed_objects()

        input_centroids = np.array([d[0] for d in detections])

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
            return self.get_confirmed_objects()

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        D = dist.cdist(np.array(objectCentroids), input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows, usedCols = set(), set()

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = input_centroids[col]
            self.disappeared[objectID] = 0
            _, radius, class_id, confidence = detections[col]
            self.class_history[objectID].append(class_id)
            self.confidence_history[objectID].append(confidence)
            self.radius_history[objectID].append(radius)
            self.hits[objectID] += 1
            if self.hits[objectID] >= self.confirm_hits:
                self.status[objectID] = 'confirmed'
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)

        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            self.hits[objectID] = 0
            self.status[objectID] = 'tentative'
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        for col in unusedCols:
            self.register(detections[col])
            
        return self.get_confirmed_objects()
