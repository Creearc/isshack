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
  socket = context.socket(zmq.REP)
  socket.bind("tcp://0.0.0.0:{}".format(5001))
  
  while True:
    msg = socket.recv()
    with lock:
      tmp = frame_tmp
    msg = pickle.dumps(tmp)
    socket.send(msg, zmq.NOBLOCK)  
      
    
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

      state, old_state = 0, 0
      buf_sec = 0
      output_file = open('train/fight_result.txt', 'w')
      
      for i in range(0, frame_count, 240):
        vid_capture.set(1, i)
        _, frame = vid_capture.read()
        if frame is None:
          continue
        
        frame, key_points = detector.detect(frame)
        state = nikita_net.run(key_points)

        cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        
        if state != old_state:
          sec = i // frame_rate
          output_file.write('{} {} {}\n'.format(buf_sec, sec, old_state))
          buf_sec = sec
        old_state = state
        
        with lock:
          frame_tmp = frame.copy()
          
      with lock:
        video_name = None
        
      output_file.close()
        
    time.sleep(0.02)


if __name__ == '__main__':
  FOLDER_PATH = 'tmp'
  lock = threading.Lock()

  video_name = None
  video_name = '{}/fight_train.mp4'.format(FOLDER_PATH)

  frame_tmp = None
  
  threading.Thread(target=main_thread, args=()).start()
  threading.Thread(target=server_thread, args=()).start()