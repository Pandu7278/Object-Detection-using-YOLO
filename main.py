import numpy as np
import argparse
import imutils
import time
import cv2
import os
from datetime import datetime
import json

#from gtts import gTTS
import os
#from playsound import playsound

language = 'en'

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
      dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolo.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
(centerX, centerY, width, height) =box.astype("int")

x = int(centerX - (width / 2))
y = int(centerY - (height / 2))

boxes.append([x, y, int(width), int(height)])
confidences.append(float(confidence))
classIDs.append(classID)
print(confidences)
print(["threshold"])
idxs = cv2.dnn.NMSBoxes(boxes, confidences,
args["confidence"],args["threshold"])
# ensure at least one detection exists
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
(w, h) = (boxes[i][2], boxes[i][3])
color = [int(c) for c in COLORS[classIDs[i]]]
cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
text = "{}:{:.4f}".format(LABELS[classIDs[i]],confidences[i])
cv2.putText(frame, text, (x, y -
5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#myobj = gTTS(text=LABELS[classIDs[i]], lang=language,
slow=False
#date_string = datetime.now().strftime("%d%m%Y%H%M%S")
#filename = "voice"+date_string+".mp3"
#myobj.save(filename)
#playsound(filename)
#os.remove(filename)
# display frame
cv2.imshow('frame', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):breakpoint()
