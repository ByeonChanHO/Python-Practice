import numpy as np
import cv2

def showVideo():
    try :
        #cap = cv2.VideoCapture(0)
        #숫자는 0번 카메라 사용
        #VideoCapture 인덱스는 장치인덱스 또는 비디오파일 이름이 옵니다.
        #장치인덱스는 어떤 카메라를 지정할 것인지에 대한 숫자이다.
        cap = cv2.VideoCapture('Read_Video_File.gif')
        #영상 위치 지정 하기
    except:
        return
    cap.set(3, 480) #3번은 가로 길이로 480을 설정
    cap.set(4, 320) #4번은 세로 길이로 320을 설정

    while True :
        #라이브로 비디오가 들어오기때문에 프레임 별로 캡쳐해주고 계속해서 디스플레이 해주어야하기에 무한 반복으로 만든다.
        ret, frame = cap.read()
        #비디오의 한 프레임을 읽는다 읽는데 성공하면 ret에 true가 들어오고 실패하면 false 가 들어온다.
        #읽는 프레임은 frame 에 값이 들어가지게 된다.

        if not ret:
            break
        #프레임 읽기가 실패하면 if문으로 무한 루프인 while문을 빠져나온다

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #읽은 프레임을 흑백으로 바꾼다.
        cv2.imshow('video', gray)
        #흑백으로 변한 프레임을 만들어진 video 창에 띄운다.

        k = cv2.waitKey(1) & 0xFF
        #이벤트가 들어올때까지 1ms 기다린다 아무 버튼을 누를시 그 번튼의 값이 리턴덴다.
        if k == 27 :
            break
        # ESC 가 눌릴때 동영상은 종료된다.

    cap.release() #종료
    cv2.destroyAllWindows() #window 창 종료

showVideo()