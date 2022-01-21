import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video Feed
video_path = 'D:/hackathon/ISShack/train_dataset/fight_train.mp4'
image_path = 'D:/hackathon/ISShack/train_dataset/6.jpg'

if_image = not True

with mp_pose.Pose(#static_image_mode=True,
                  model_complexity=1,
                  smooth_landmarks=True,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5
                  ) as pose:
    if if_image:
        image = cv2.imread(image_path)
        
        # Recolor image to RGB
        #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection with MediaPipe
        #startT = time.time()
        results = pose.process(image)
        #print(time.time()-startT)

        # Recolor image to BGR
        image.flags.writeable = True
        #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #print(results)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Draw detected landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  #mp_drawing.DrawingSpec(color=(245,117,66), thinckness=2, circle_radius=2),
                                  #mp_drawing.DrawingSpec(color=(245,66,230), thinckness=2, circle_radius=2),
                                 )

        # Show Feed
        cv2.imshow('feed', image)
        cv2.waitKey(0) 

    else:
        cap = cv2.VideoCapture(video_path)
        pTime = 0
        cTime = 0
        sTime = 0
        while cap.isOpened():
            ret, frame = cap.read()

            # Resize frame
            frame = cv2.resize(frame, (1280,720))
            
            # Recolor image to RGB
            #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame
            image.flags.writeable = False

            # Pose detection with MediaPipe
            #startT = time.time()
            results = pose.process(image)
            #print(time.time()-startT)

            # Recolor image to BGR
            image.flags.writeable = True
            #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #print(results)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # Draw detected landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      #mp_drawing.DrawingSpec(color=(245,117,66), thinckness=2, circle_radius=2),
                                      #mp_drawing.DrawingSpec(color=(245,66,230), thinckness=2, circle_radius=2),
                                     )
            #print(results.pose_world_landmarks)

            

            # Calculate FPS and put it in the image
            cTime = time.time()
            TIME = time.time() - sTime
            if TIME >= 1:
                fps = 1/(cTime-pTime)
                sTime = time.time()
            pTime = cTime

            cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

            # Show Feed
            cv2.imshow('feed', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()   
cv2.destroyAllWindows()