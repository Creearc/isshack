import cv2
import mediapipe as mp
import numpy as np
import time
import math

class poseDetector():
    
    def __init__(self, mode=False, 
                 smooth=True,
                 modelComplexity=1, 
                 detectionCon=0.5, trackingCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode,
                                      model_complexity=self.modelComplexity,
                                      smooth_landmarks=self.smooth,
                                      min_detection_confidence=self.detectionCon,
                                      min_tracking_confidence=self.trackingCon)



    def findPose(self, image, draw=True, circle_radius=2):
        
        # Recolor image to RGB
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        thickness = (image.shape[0] + image.shape[1]) // 600
        image.flags.writeable = False

        # Pose detection with MediaPipe
        self.results = self.pose.process(image)

        # Recolor image to BGR
        image.flags.writeable = True
        #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw detected landmarks
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, 
                                               self.mp_pose.POSE_CONNECTIONS, 
                                               #self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=thickness, circle_radius=circle_radius),
                                               #self.mp_drawing.DrawingSpec(color=(255,0,255), thickness=thickness, circle_radius=circle_radius),
                                              )
        return image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS


    def drawPose(self, image, circle_radius=1):
        
        thickness = (image.shape[0] + image.shape[1]) // 600
        
        # Draw detected landmarks
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, 
                                           self.mp_pose.POSE_CONNECTIONS, 
                                           self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=thickness, circle_radius=circle_radius),
                                           self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=thickness, circle_radius=circle_radius),
                                          )
        return image

    def findPosition(self, image, draw=True):
        self.lmList = []
        #lmListAllAtr = []
        fontScale = (image.shape[0] + image.shape[1]) // 500
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy,round(lm.visibility, 2)])
                if draw:
                    cv2.circle(image, (cx,cy),fontScale,(255,0,0),cv2.FILLED)

        return self.lmList


    def recalculate_lm(self, image, draw=False):
        self.rc_lmList = []
        fontScale = (image.shape[0] + image.shape[1]) // 500
        if len(self.lmList) != 0:
            middle_point = (int((self.lmList[25][1]+self.lmList[24][1])/2), 
                            int((self.lmList[25][2]+self.lmList[24][2])/2))


        for kp in self.lmList:
            id, new_cx, new_cy = kp[0] , kp[1]-middle_point[0], kp[2]-middle_point[1]
            self.rc_lmList.append([new_cx, new_cy])
            #self.rc_lmList.append([id, new_cx, new_cy])
            if draw:
                cv2.circle(image, (new_cx+middle_point[0], new_cy+middle_point[1]),fontScale,(0,255,0),cv2.FILLED)

        self.coords = []
        for elem in self.rc_lmList:
            for coord in elem:
                 self.coords.append(coord)

        return self.coords


def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


if __name__ == "__main__":
    import pprint

    # Video Feed
    vid_path = 'D:/exercise_tracker/kio/WhatsApp_Video_2022-02-18_at_11_23_05.mp4'
    cap = cv2.VideoCapture(vid_path)
    detector = poseDetector(modelComplexity=1)

    while cap.isOpened():

        ret, image = cap.read()
        image = rotate_img(image, 270)

        # Resize image
        #image = cv2.resize(image, (720,405))

        image, _, _ = detector.findPose(image, draw=False)
        lmList = detector.findPosition(image, draw=False)
        rc_lmList = detector.recalculate_lm(image, draw=True)

        print(len(rc_lmList))
        # Show Feed
        cv2.imshow('feed', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
