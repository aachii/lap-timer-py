#!/usr/bin/env python

import cv2
import time
import sys
import numpy as np
from scipy.spatial import distance as dist
import time


class BlobTracker:
    def __init__(self, maxDisappeared=120, histThreshold=0.00005, distThreshold=1000):
        self.nextID = 1
        self.objects = {}          # id -> centroid
        self.histograms = {}       # id -> color histogram
        self.disappeared = {}      # id -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.histThreshold = histThreshold
        self.distThreshold = distThreshold

    def _compute_hist(self, frame, x, y, w, h):
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180, 0,256])
        cv2.normalize(hist, hist)
        return hist

    def register(self, centroid, hist):
        self.objects[self.nextID] = centroid
        self.histograms[self.nextID] = hist
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.histograms[objectID]
        del self.disappeared[objectID]

    def update(self, frame, detections):
        # detections = list of (x, y, w, h, cx, cy)

        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # compute new centroids + histograms
        newCentroids = []
        newHists = []
        for (x, y, w, h, cx, cy) in detections:
            newCentroids.append((cx, cy))
            newHists.append(self._compute_hist(frame, x, y, w, h))

        # first frame
        if len(self.objects) == 0:
            for c, h in zip(newCentroids, newHists):
                self.register(c, h)
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        objectHists = [self.histograms[i] for i in objectIDs]

        # centroid distance matrix
        D = dist.cdist(np.array(objectCentroids), np.array(newCentroids))

        # histogram similarity matrix
        H = np.zeros((len(objectIDs), len(newCentroids)))
        for i, histA in enumerate(objectHists):
            for j, histB in enumerate(newHists):
                H[i, j] = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

        # combine: high hist + low distance
        D_norm = D / (np.max(D) + 1e-6)
        score = H - D_norm

        rows = score.max(axis=1).argsort()[::-1]
        cols = score.argmax(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if H[row, col] < self.histThreshold:
                continue
            if D[row, col] > self.distThreshold:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = newCentroids[col]
            self.histograms[objectID] = newHists[col]
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # disappeared objects
        for row in set(range(len(objectIDs))) - usedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        # new objects
        for col in set(range(len(newCentroids))) - usedCols:
            self.register(newCentroids[col], newHists[col])

        return self.objects


def formatTime(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


startTime = time.time() 

#video = 'mini-z-30fps-cfr-noaudio-cut.mp4'
#cap = cv2.VideoCapture(video)
camIndex = 0
cap = cv2.VideoCapture(0)

fps = 30
sleepTime = 1 / fps
speed = 1

lineY = 0
def mouseDown(event, x, y, flags, param):
    global lineY
    if event == cv2.EVENT_LBUTTONDOWN:
        lineY = y

ret, frame = cap.read()
if not ret:
    sys.exit()

# frame is a numpy array
height, width, color = frame.shape

scaleH = (height*2)/3
scaleW = (width*2)/3
minArea = (scaleH / 6) * ( scaleW / 6)
maxArea = (scaleH / 3) * ( scaleW / 3)

frame = cv2.resize(frame, dsize=(int(scaleW), int(scaleH)), interpolation=cv2.INTER_CUBIC)

# openCV stuff
backsub = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400, detectShadows=True)

tracker = BlobTracker()
lapStart = {}
prevY = {}   # <-- added for direction-aware crossing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame is a numpy array
    frame = cv2.resize(frame, dsize=(int(scaleW), int(scaleH)), interpolation=cv2.INTER_CUBIC)

    # openCV stuff
    fgmask = backsub.apply(frame)

    # remove shadows
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    detections = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < minArea or area > maxArea:
            continue

        cx, cy = centroids[i]
        detections.append((x, y, w, h, int(cx), int(cy)))
    
    objects = tracker.update(frame, detections)

    # draw detections and IDs
    for (x, y, w, h, cx, cy) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    currentTime = time.time() - startTime

    for objectID, (cx, cy) in objects.items():
        cv2.putText(frame, f"ID {objectID}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.circle(frame, (cx, cy), 5, (0,255,255), -1)

        # direction-aware lap logic
        if objectID not in prevY:
            prevY[objectID] = cy
        else:
            crossed = False

            if prevY[objectID] < lineY and cy >= lineY:
                crossed = True
            if prevY[objectID] > lineY and cy <= lineY:
                crossed = True

            if crossed:
                if objectID not in lapStart:
                    lapStart[objectID] = currentTime
                else:
                    lapTime = currentTime - lapStart[objectID]
                    print(f"ID {objectID} LAP: {formatTime(lapTime)}")
                    lapStart[objectID] = currentTime

            prevY[objectID] = cy

    # finish line
    cv2.line(frame, (0, lineY), (int(scaleW), lineY), (0,0,255), 5)

    # global timer
    timerText = formatTime(currentTime)
    cv2.putText(frame, timerText, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", fgmask)
    cv2.setMouseCallback("Frame", mouseDown)

    time.sleep(sleepTime * (1 / speed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
