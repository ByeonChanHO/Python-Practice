import numpy as np
import cv2

def showImage() :
    imgfile = 'Read_image_File.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    cv2.imshow('Pear', img)
    '''타이틀 Pear 이며 img은 파일 읽은 것의 리턴값'''
    cv2.waitKey(0)
    '''키 입력을 기다리는 대기함수 매개변수 만큼 기다리고 다음으로 넘어감 0은 무한대'''
    cv2.destroyAllWindows()
    '''화면에 나타난 윈도우 종료'''

showImage()