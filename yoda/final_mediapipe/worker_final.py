import threading
import time
import zmq
import numpy as np
import cv2
import pickle


from modules import pose_estimation_mp_module as dt
from modules import nikita_net

import zmq_module
     
    
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
      output_file = open('fight_result.txt', 'w')

      per, old_per = 0, -1

      first_frame = 0
      last_frame = frame_count
      step = 1
      
      for i in range(first_frame, last_frame, step):
        per = int(i / frame_count * 100)
        if per != old_per:
          print('{}%'.format(per))
          old_per = per
        vid_capture.set(1, i)
        _, frame = vid_capture.read()
        if frame is None:
          continue

        frame = dt.rotate_img(frame, 270)
        frame, _, _ = detector.findPose(frame, draw=False)
        lmList = detector.findPosition(frame, draw=False)
        key_points = detector.recalculate_lm(frame, draw=True)

        if key_points is None or len(key_points) == 0:
          continue        
        state = nikita_net.run(key_points)
        print(state)

        cv2.putText(frame, '{}'.format(state), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                      (0, 0, 255), 2)

 
        
        if state != old_state and i // frame_rate > 0 or i + step >= last_frame:
          sec = i // frame_rate
          output_file.write('{} {} {}\n'.format(buf_sec, sec, old_state))
          buf_sec = sec
        old_state = state
        
        serv.put_img(frame)
          
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
  detector = dt.poseDetector(modelComplexity=1)

  video_name = None
  video_name = '../WhatsApp_Video_2022-02-18_at_11_23_05.mp4'

  frame_tmp = None
  
  threading.Thread(target=main_thread, args=()).start()

  serv = zmq_module.ZMQ_transfer('0.0.0.0', 5005)
  serv.run()
  
