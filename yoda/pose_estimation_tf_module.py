import tensorflow as tf
import tensorflow_hub as hub
import cv2
#from matplotlib import pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
import pprint
import functions

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

def detect(image):
    image = cv2.resize(image, (1280,768))
    img = image.copy()

    #img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192,320)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    new_kp_frame = data_prep(keypoints_with_scores, 0.2)
    #print(keypoints_with_scores)
    
    # Render keypoints 
    loop_through_people(image, keypoints_with_scores, EDGES, 0.2)

    return image, new_kp_frame

def data_prep(keypoints_with_scores, min_conf):
    new_kp_frame = []
    
    for person in keypoints_with_scores:
        mean_conf = 0
        people = []
        for kp in person:
            ky, kx, kp_conf = kp
            mean_conf = mean_conf + kp_conf
            people.append([ky, kx]) 
        mean_conf = mean_conf/len(person)
        if mean_conf > min_conf:
            new_kp_frame.append(people)
    
    if new_kp_frame == []: return None
    
    data = functions.padding(new_kp_frame, size=10)
    return data
        


if __name__ == '__main__':

    # optional if you are using a GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # load MoveNet model
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    # set video path
    video_path = 'WIN_20220218_10_07_32_Pro.mp4'
    cap = cv2.VideoCapture(video_path)
    # chose start frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(0, frame_count, 1):

        cap.set(1, i)
        _, frame = cap.read()
        if frame is None:
            print('Err: frame is None')
            continue

        frame, new_kp_frame = detect(frame)
        #pprint.pprint(new_kp_frame)
        
        # Render keypoints 
        cv2.imshow('Movenet Multipose', frame)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
