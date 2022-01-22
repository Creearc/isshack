import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
#import mediapipe as mp
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import pandas as pd
import pickle

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

def shift(data):
  out = []
  out.append(data)
  sec, people = data
  for i in range(1, len(people)):
    start = people[len(people) - i: ]
    end = people[ : len(people) - i]
    out.append([sec, start + end])
  
  return out

def padding(data, size=10):
  sec, people = data
  start_size = len(people)
  for i in range(size - start_size):
    people.append(people[i])
  return [sec, people]

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

video_path = '../tmp/fight_train.mp4'
if_image = not True

#res_line = {'sec':[], 'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10': []}
#resList = pd.DataFrame(res_line)

#line_count = 0

cap = cv2.VideoCapture(video_path)
# chose start frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sec = 0
out = []
#for i in range(0, 4800, 240):
for i in range(0, frame_count, 240):
    if i % 100 == 0:
        print('{}%'.format(int(i / frame_count * 100)))

    cap.set(1, i)
    _, frame = cap.read()
    if frame is None:
        continue

    frame = cv2.resize(frame, (1280,768))
    
    # Resize image
    img = frame.copy()

    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192,320)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)

    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    new_kp_frame = []

    for person in keypoints_with_scores:
        mean_conf = 0
        for kp in person:
            ky, kx, kp_conf = kp
            mean_conf = mean_conf + kp_conf
            #np.append(people, [ky, kx]) 
        mean_conf = mean_conf/len(person)
        if mean_conf > 0.2:
            new_kp_frame.append(person)

    new_kp_frame = [sec, new_kp_frame]
    sec += 1

    #print(new_kp_frame)
    if new_kp_frame[1] == []: continue

    data = padding(new_kp_frame, size=10)
    
    #print(data)

    data = shift(data)
    #print(data)

    out = out + data 
    '''
    for fr in data:
        print(fr)
        sec, peop = fr
        new_row = pd.DataFrame({'sec': sec, 
                                'p1': peop[0], 'p2': peop[1], 'p3': peop[2], 
                                'p4': peop[3], 'p5': peop[4], 'p6': peop[5], 
                                'p7': peop[6], 'p8': peop[7], 
                                'p9': peop[8], 'p10': peop[9]}, index = [line_count])
        resList = pd.concat([resList, new_row]).reset_index(drop = True)
        line_count += 1
    '''

    
    # Render keypoints 
    #loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)
    #cv2.imshow('Movenet Multipose', frame)
    
    #if cv2.waitKey(1) & 0xFF==ord('q'):
    #    break

if out != []:
    with open('data.pickle', 'wb') as f:
        pickle.dump(out, f)
else:
    print('something bad has happened')

#print(resultList)
#resultList.to_csv("res.csv")

cap.release()
cv2.destroyAllWindows()