import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pickle
import pprint

import pose_estimation_mp_module as dt


def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


if __name__ == "__main__":
    # Video Feed
    vid_path = '../WhatsApp_Video_2022-02-18_at_11_23_05.mp4'
    cap = cv2.VideoCapture(vid_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = dt.poseDetector(modelComplexity=1)

    #out = []
    out = dict()

    #while cap.isOpened():
    for i in range(0, frame_count, 1):

        ret, image = cap.read()
        image = rotate_img(image, 270)

        image, _, _ = detector.findPose(image, draw=False)
        lmList = detector.findPosition(image, draw=False)
        rc_lmList = detector.recalculate_lm(image, draw=True)

        #new_kp_frame = []
        if rc_lmList == []: continue
        #new_kp_frame = [i, rc_lmList]
        #out.append(new_kp_frame) 
        
        out[i] = rc_lmList

        # Show Feed
        cv2.imshow('feed', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    #if out != []:
    if len(out.keys()) > 0:
        with open('data.pickle', 'wb') as f:
            pickle.dump(out, f)
        print('saved')
    else:
        print('something bad has happened')

    pprint.pprint(out)    

    cap.release()
    cv2.destroyAllWindows()
