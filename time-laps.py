#!/usr/bin/env python

import cv2
import time
import sys
import numpy as np

#video = 'mini-z-30fps-cfr-noaudio-cut.mp4'
#cap = cv2.VideoCapture(video)
camIndex = 0
cap = cv2.VideoCapture(0)

fps = 60
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
height = None
width = None
color = None
if height is None and width is None:
    height, width, color = frame.shape

scaleH = (height*2)/3
scaleW = (width*2)/3
minArea = (scaleH / 6) * ( scaleW / 6)
maxArea = (scaleH / 3) * ( scaleW / 3)

frame = cv2.resize(frame, dsize=(int(scaleW), int(scaleH)), interpolation=cv2.INTER_CUBIC)

# openCV stuff
backsub = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400, detectShadows=True)

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

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < minArea or area > maxArea:
            continue

        cx, cy = centroids[i]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

    cv2.line(frame, (0, lineY), (int(scaleW), lineY), (0,0,255), 5)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", fgmask)
    cv2.setMouseCallback("Frame", mouseDown)

    time.sleep(sleepTime * (1 / speed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()