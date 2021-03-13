import numpy as np
import cv2

def drawing():
    img = np.zeros((512, 512, 3), np.uint8)
    #그리는 공간을 만들어야하기에 512x512 를 만들어 1칸당 (0,0,0) 을 가지게 만든다. 0~255
    #np.zeros((가로, 세로, 한 픽셀당 가지는 배열 수), 타입)

    cv2.line(img, (00,00), (511,511), (0,255,0), 5)
    #img 공간에서 선을 만든다
    #cv2.line(공간, 시작지점, 끝나는 지점, BGR색깔, 굵기)
    cv2.rectangle(img, (384, 0), (510,128), (0,0,255), 3)
    #네모칸을 만든다.
    #cv2.rectangle(공간, 시작지점, 끝지점, BGR 색깔, 굵기)
    cv2.circle(img, (447, 63), 63, (255, 0, 0), -1)
    #원을 만든다.
    #cv2.circle(공간, 원의 중심, 반지름, BGR색깔, 원의 굵기) 원의 굵기가 -1 일경우 안의 색이 채워집니다.
    cv2.ellipse(img, (256, 256), (120, 40), 0, -100, 180, (255, 0, 0), -1)
    #타원을 만든다
    #cv2.ellipse(공간, 타원의 중심좌표, (장축, 단축), 타원의 기울기 각도, 타원의 호를 그리는 시작 각도, 타원의 호를 그리는 끝 각도, BGR 색상, 굵기)
    #굵기가 -1 일 경우 도형 안을 채웁니다.

    font = cv2.FONT_HERSHEY_SIMPLEX
    #글자 스타일
    cv2.putText(img, 'Chan', (10, 500), font, 5, (255, 255, 255), 2)
    #글자 삽입
    #cv2.putText(공간, 넣을 글자, 시작 좌표, 글자 스타일, 글자 크기, BGR 색상, 굵기)

    cv2.imshow('drawing',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

drawing()