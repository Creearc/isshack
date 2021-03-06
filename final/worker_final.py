import threading
import time
import zmq
import numpy as np
import cv2
import pickle

import pose_estimation_tf_module as detector
import nikita_net

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
      w = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      print('Go!')

      state, old_state = 0, 0
      buf_sec = 0
      output_file = open('results/fight_result.txt', 'w')

      per, old_per = 0, -1

      first_frame = 0
      last_frame = frame_count
      step = frame_rate
      
      for i in range(first_frame, last_frame, step):
        per = int(i / frame_count * 100)
        if per != old_per:
          print('{}%'.format(per))
          old_per = per
        vid_capture.set(1, i)
        _, frame = vid_capture.read()
        if frame is None:
          continue
        
        frame, key_points = detector.detect(frame)
        if key_points is None:
          continue        
        state = nikita_net.run(key_points)

        if state == 1:
          cv2.putText(frame, 'ALARM!', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                      (0, 0, 255), 2)

          cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                        (0, 0, 255), 25)
        
        if state != old_state and i // frame_rate > 0 or i + step >= last_frame:
          sec = i // frame_rate
          output_file.write('{} {} {}\n'.format(buf_sec, sec, old_state))
          buf_sec = sec
        old_state = state
        
        with lock:
          frame_tmp = frame.copy()
          
      with lock:
        video_name = None

      
      sec = frame_count // frame_rate
      output_file.write('{} {} {}\n'.format(buf_sec, sec, state))
           
      output_file.close()
      print('Done!')
        
    time.sleep(0.02)


if __name__ == '__main__':
  FOLDER_PATH = '../train'
  lock = threading.Lock()

  video_name = None
  video_name = '{}/fight_test.mp4'.format(FOLDER_PATH)

  frame_tmp = None
  
  threading.Thread(target=main_thread, args=()).start()
  threading.Thread(target=server_thread, args=()).start()
