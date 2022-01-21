import threading
import time
import zmq
import numpy as np
import cv2
import pickle

from key_point import pose_detection_1 as detector


def server_thread():
  global video_name, frame_tmp, lock

  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.bind("tcp://0.0.0.0:{}".format(5001))
  
  while True:
    with lock:
      tmp = frame_tmp
    msg = pickle.dumps(tmp)
    socket.send(10001, msg)  
      
    
def main_thread():
  global video_name, frame_tmp, lock
  while True:
    with lock:
      v = video_name
    if not (v is None):
      vid_capture = cv2.VideoCapture(v)
      frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
      frame_rate = int(vid_capture.get(cv2.CAP_PROP_FPS))
      print(frame_count, frame_rate)
      w = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      print('Go!')

      for i in range(0, frame_count, 1):
        vid_capture.set(1, i)
        _, frame = vid_capture.read()
        if frame is None:
          continue
        
        frame = detector.detect(frame)
        with lock:
          frame_tmp = frame.copy()
      with lock:
        video_name = None
        
    time.sleep(0.02)


if __name__ == '__main__':
  FOLDER_PATH = 'tmp'
  lock = threading.Lock()

  video_name = None
  video_name = '{}/fight_train.mp4'.format(FOLDER_PATH)

  frame_tmp = None
  
  threading.Thread(target=main_thread, args=()).start()
  threading.Thread(target=server_thread, args=()).start()
