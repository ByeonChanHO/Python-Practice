import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

def faceDetect():
    eye_detect = True
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontface.xml')
    # 얼굴을 학습 시킨다.
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    # 눈을 학습시킨다.

    info = ''
    try :
        cap = cv2.VideoCapture(0)
        #0번 카메라 사용
    except:
        print('cannot load cam')
        return

    while True:
        ret, frame = cap.read()
        if eye_detect:
            info = 'Eye Ddetection On'
        else :
            info = 'Eye Detection off'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #회색 이미지로 저장
        faces = face_cascade.detectMultiScale(gray,1.3,5) #얼굴 인식
        cv2.putText(frame,info, (5,15), font, 0.5, (255,0,255), 1) #화면 오른쪽 상단 글자를 붙임

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y),(x + w, y+h), (255, 0, 0), 2) #얼굴 인식한 곳 표시
            cv2.putText(frame, 'Detected Face', (x - 5, y-5), font, 0.5, (255, 255, 0), 2) #얼굴 인식한 곳에 텍스트 붙임
            if eye_detect:
                roi_gray = gray[y:y + h, x: x + w] #얼굴인식한 부분 회색색상으로 저장
                roi_color = frame[y:y + h, x: x + w] #얼굴인식한 부분 컬러색상으로 저장
                eyes = eye_cascade.detectMultiScale(roi_gray) #눈인식
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey+eh), (0, 255,0), 2) #눈 인식한 부분 표신
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) #30ms동안 입력을 기다린다.
        if k == ord('i'): #입력이 i가 들어올경우
            eye_detect = not eye_detect #눈 인식 종료
        if k == 27: #ESC 면 종료
            break
    cap.release()
    cv2.destroyAllWindows()
faceDetect()
