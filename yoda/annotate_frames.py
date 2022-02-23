import cv2
import numpy as np

class Video():
    def __init__(self, path, use_buffer=True):
        self.path = path
        self.cam = cv2.VideoCapture(self.path)
        self.frame_count = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cam.get(cv2.CAP_PROP_FPS))
        self.w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def stop(self):
        self.cam.release()        


    def get_img(self, first_frame=0, last_frame=-1, step=1):
        last_frame = self.frame_count if last_frame == -1 else last_frame
        for frame_number in range(first_frame, last_frame, step):
            ret, img = self.cam.read()
            if not ret:
                time.sleep(0.1)
                continue
            yield frame_number, img


def sec_to_frame_number(sec, frame_rate):
  frame_number = sec * frame_rate
  return frame_number


if __name__ == '__main__':

  c = Video('8.mp4')

  arr = [
    (sec_to_frame_number(5, c.frame_rate),
     sec_to_frame_number(12, c.frame_rate), 'FUDO_DACHI'),
    
    (sec_to_frame_number(17, c.frame_rate),
     sec_to_frame_number(20, c.frame_rate), 'ZENKUTSU_DACHI'),
    
    (sec_to_frame_number(47, c.frame_rate),
     sec_to_frame_number(50, c.frame_rate), 'ZENKUTSU_DACHI'),
    
    (sec_to_frame_number(53, c.frame_rate),
     sec_to_frame_number(56, c.frame_rate), 'FUDO_DACHI'),
    
    (sec_to_frame_number(59, c.frame_rate),
     sec_to_frame_number(61, c.frame_rate), 'ZENKUTSU_DACHI'),
    
    (sec_to_frame_number(74, c.frame_rate),
     sec_to_frame_number(80, c.frame_rate), 'ZENKUTSU_DACHI'),

    (sec_to_frame_number(98, c.frame_rate),
     sec_to_frame_number(99, c.frame_rate), 'ZENKUTSU_DACHI'),
    
    (sec_to_frame_number(105, c.frame_rate),
     sec_to_frame_number(108, c.frame_rate), 'FUDO_DACHI'),
    ]
  

  f = open('annotaion.txt', 'w')
  interval = 0
  for i in range(c.frame_count):
    if i >= arr[interval][0] and i <= arr[interval][1]:      
      cl = arr[interval][2]
      if i == arr[interval][1] and interval < len(arr) - 1:
        interval += 1
    else:
      cl = 'None'
    f.write('{} {}\n'.format(i, cl))
  f.close()
    
        
