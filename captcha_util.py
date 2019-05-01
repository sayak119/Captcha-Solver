import cv2 as cv
import numpy as np
import argparse
import subprocess
import time
import os


# label_file = "data/detect/lpr_detect.names"
# config_file = "data/detect/yolov3-tiny-lpr-detect.cfg"
# weight_file = "data/detect/yolov3-tiny-lpr-detect_36100.weights"

label_file = "data/captcha.names"
# config_file = "data/yolov3-tiny-captcha.cfg"
# weight_file = "data/yolov3-tiny-captcha_last.weights"
config_file = "data/yolov3_captcha.cfg"
weight_file = "data/yolov3_captcha_last.weights"

g_width = 416
g_confidence = 0.5
g_threshold = 0.3

# Get the labels
g_labels = open(label_file).read().strip().split('\n')

# Intializing colors to represent each label uniquely
g_colors = np.random.randint(0, 255, size=(len(g_labels), 3), dtype='uint8')

# Load the weights and configutation to form the pretrained YOLOv3 model
g_net = cv.dnn.readNetFromDarknet(config_file, weight_file)

# Get the output layer names of the model
g_layer_names = g_net.getLayerNames()
g_layer_names = [g_layer_names[i[0] - 1] for i in g_net.getUnconnectedOutLayers()]


def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)


def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            # print (detection)
            # a = input('GO!')

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids


def infer_image(img):

    height, width = img.shape[:2]

    # Contructing a blob from the input image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (g_width, g_width), swapRB=True, crop=False)

    # Perform a forward pass of the YOLO object detector
    g_net.setInput(blob)

    # Getting the outputs from the output layers
    start = time.time()
    outs = g_net.forward(g_layer_names)
    end = time.time()

    # Generate the boxes, confidences, and classIDs
    boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, g_confidence)

    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, g_confidence, g_threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        return None, None, None, None, None

    ret_boxes = []
    ret_confidences = []
    ret_classids = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            ret_boxes.append(boxes[i])
            ret_confidences.append(confidences[i])
            ret_classids.append(classids[i])

    # Draw labels and boxes on the image
    # img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, g_colors, g_labels)

    return img, ret_boxes, ret_confidences, ret_classids
    # return boxes, confidences, classids, idxs
