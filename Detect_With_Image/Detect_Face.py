import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('./haarcascade_frontface.xml')
#얼굴을 학습 시킨다.
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
#눈을 학습시킨다.

img = cv2.imread('./face1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#색깔을 Gray로 바꾼다.
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#detectMultiScale 에서 1.3은 이미지 축적에서 크기가 얼마나 줄어들게 할지 지정하는 것
#5는 감지된 얼굴의 품질에 영향을 준다. 감지횟수와도 관련있다.

for(x,y,w,h) in faces :
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    #얼굴 감지한거 표시
    roi_gray = gray[y:y+h, x:x+w] #얼굴 감지한 부분만 그레이 색상으로 저장
    roi_color = img[y:y+h, x:x+w] #얼굴 감지한 부분만 컬러로 색상 저장
    eyes = eye_cascade.detectMultiScale(roi_gray) #눈을 감지한다.
    for(ex,ey,ew,eh) in eyes :
        cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0), 2) #눈 부분 표시

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()