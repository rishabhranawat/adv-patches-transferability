import cv2 as cv
import numpy as np
import time

img = cv.imread('/home/rdr2143/waymo-adv-dataset/adv/patched/0_patched_img.jpg')
classes = open('/home/rdr2143/darknet/data/coco.names').read().strip().split('\n')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('/home/rdr2143/darknet/cfg/yolov2.cfg', '/home/rdr2143/darknet/yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]

net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()

for out in outputs:
    print(out[0].shape)

# for output in outputs:
#     for detection in output:
#         scores = detection[5:]
#         classID = np.argmax(scores)
#         confidence = scores[classID]
#         if confidence > 0.5:
#             print(classes[classID])