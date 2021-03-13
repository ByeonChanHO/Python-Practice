import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontface.xml') #얼굴 학습
face_mask = cv2.imread('./pororo.jpg') #마스크로 쓸 이미지
h_mask, w_mask = face_mask.shape[:2]

if face_cascade.empty(): #얼굴 학습 안될시
    raise IOError('Unable to load the face cascade classifier xml file')
cap = cv2.VideoCapture(0) # 0번 카메라 호출
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #회색화면으로 저장
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5) #얼굴 인식

    for(x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            x = int(x-w*0.1 )
            y = int(y -h*0.05)
            w = int(1.2* w)
            h = int(1.2 * h)
            frame_roi = frame[y:y + h, x:x + w]
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
            #영역 보간법을 써서 이미지 사이즈를 조정한다.
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY_INV)
            #이미지를 global 하게 흑백으로 저장.
        mask_inv = cv2.bitwise_not(mask)
        #흑백을 반대로
        masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)
        #카메라에 얼굴 부분에 마스크 이미지 적용
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()