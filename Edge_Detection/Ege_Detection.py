import numpy as np
import cv2

def showEdge():
    imgfile = 'Read_image_File.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    #이미지 Grayscale 로 읽는다.

    edge1 = cv2.Canny(img, 30, 200)
    edge2 = cv2.Canny(img, 180, 200)
    #Canny Edge Detection 알고리즘을 구현한 것
    #두번째 인자는 제일 작은 threshoding 값을 넣고
    #세번째 인자는 제일 큰 threshoding 값을 넣는다.

    cv2.imshow('canny1', edge1)
    cv2.imshow('canny2', edge2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

showEdge()
