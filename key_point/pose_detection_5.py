import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

##net = cv2.dnn.readNetFromDarknet('key_point/model_5/yolov4.cfg',
##                                 'key_point/model_5/yolov4.weights')
net = cv2.dnn.readNetFromDarknet('key_point/model_5_1/yolov4-sam-mish.cfg',
                                 'key_point/model_5_1/yolov4-sam-mish.weights')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

scale = 0.00392

conf_threshold = 0.5
nms_threshold = 0.4


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(class_id)
    out = img[y : y_plus_h, x : x_plus_w].copy()
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (255, 255, 255), 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img



def detect(img):
    (Height, Width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        # i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        img = draw_prediction(img, class_id, confidences[i],
                              round(x), round(y), round(x + w), round(y + h))

            
    return img
    

