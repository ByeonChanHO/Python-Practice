import numpy as np
import cv2

def showGreen():
    imgfile = 'Read_image_File.jpg' #이미지 파일을 가져오기
    img = cv2.imread(imgfile, cv2. IMREAD_COLOR) # 이미지 파일 컬러로 읽기
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #기본 BGR을 HSV로 바꾼다.

    lower_red = np.array([28, 34, 100])
    #하한값 설정
    upper_red = np.array([60, 255, 255])
    #상한값 설정

    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    #cv2.inRange(검사할 이미지, 하한값, 상향값)
    #각 픽셀을 검사하는데 하한값과 상한값 사이에 들어오면 값이 흰색(255)이고 그렇지 않으면 검은색(0)을 표시한다.
    res = cv2.bitwise_and(img, img, mask=mask_red)
    #bitwise_and는 첫번째 인자와 두번째 인자를 and연산하여 표시하는 데 이때 세번째 인자인 mask 값이 0인 부분이 있는 공간이면 검은색 그대로 0값으로 만듭니다.

    cv2.imshow('apple_red', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
showGreen()
