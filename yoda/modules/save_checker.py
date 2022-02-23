import cv2
import pickle


def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def draw_lm(image, lm, draw=True):
    fontScale = (image.shape[0] + image.shape[1]) // 500
    
    middle_point = [100, 350]
    
    for point in lm:
        if draw:
            cv2.circle(image, (point[0]+middle_point[0], point[1]+middle_point[1]),
                                                fontScale,(255,0,0),cv2.FILLED)
    #return image


with open('data.pickle', 'rb') as f:
    kp = pickle.load(f)


vid_path = '../WhatsApp_Video_2022-02-18_at_11_23_05.mp4'
cap = cv2.VideoCapture(vid_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(0, frame_count, 1):

    ret, image = cap.read()
    image = rotate_img(image, 270)

    if i in kp.keys():
        lm = kp[i]
        draw_lm(image, lm)

     # Show Feed
    cv2.imshow('feed', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
