import cv2

# Video Feed
video_path = 'D:/hackathon/ISShack/train_dataset/fight_train.mp4'
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    # Resize frame
    frame = cv2.resize(frame, (1280,768))

    cv2.imshow('feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()