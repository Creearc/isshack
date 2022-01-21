import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
#import mediapipe as mp
import time

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 3, (0,0,255), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

image_path = 'D:/hackathon/ISShack/train_dataset/6.jpg'
video_path = 'D:/hackathon/ISShack/train_dataset/fight_train.mp4'
if_image = not True

if if_image:
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (1280,768))

    img = image.copy()

    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 512,512)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    #print(keypoints_with_scores)
    
    # Render keypoints 
    loop_through_people(image, keypoints_with_scores, EDGES, 0.1)
    
    cv2.imshow('Movenet Multipose', image)
    cv2.waitKey(0) 
else:
    cap = cv2.VideoCapture(video_path)
    # chose start frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(frame_count):
        if i % 100 == 0:
            print('{}%'.format(int(i / frame_count * 100)))

        cap.set(1, i)
        _, frame = cap.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1280,768))
        
        # Resize image
        img = frame.copy()

        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # Detection section
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        #print(keypoints_with_scores)
        
        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
        
        cv2.imshow('Movenet Multipose', frame)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()