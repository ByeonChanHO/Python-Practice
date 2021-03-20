# References
# https://deep-eye.tistory.com/6
# https://bong-sik.tistory.com/16

#영상 불러오기 모델을 불러오고 하나의 비디오 각 frame을 추출하고
# 각 frame 마다 img <- frame 을 해서 그 한 frame 당 d<-detect(img) 한다.
#detect가 된다면 이제 시각화 작업을 해야한다.
#visualize(d,img)
#그리고 영상 종료 끄기한다.

#물체를 받아오는 확률이 0.5보다 높으면 confident를 뜻 물체라 인식
#Yolo은 하나의 프레임으로 작동한다.
#잡히다 안잡히는게 에러가 생긴것.
#비디오 object dectection 으로 Yolo만으로 이미지 feature을 뽑을 수 있을까?
#Yolo로 어떻게 할 수 있을까?
#리포터에서 까만 사람은 어떻게 해볼까 등등 보고서에 첨부하면 좋겠다.

import cv2
import numpy as np
import pafy

def show_YOLO_detection():
    
    # Youtube 비디오 불러오기
    url = "https://www.youtube.com/watch?v=NyLF8nHIquM" #유투브 주소
    video = pafy.new(url)
    best = video.getbest(preftype="mp4") #비디오 타입


    VideoSignal = cv2.VideoCapture()
    VideoSignal.open(best.url)

    # YOLO 가중치 파일과 CFG 파일 로드
    YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

    # YOLO NETWORK 재구성
    classes = []
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        #line.strip() = 문장의 맨앞과 맨뒤의 띄워쓰기' ', 탭'\t', 언테'\n'을 없애준다.
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cnt = 0

    while True:
        
        # 프레임 받아오기
        ret, frame = VideoSignal.read()
        frame = cv2.resize(frame, (416, 416))
        h, w, c = frame.shape

        # 성능을 위해 프레임 생략
        cnt += 1
        if cnt % 4 != 0:
            continue

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        #cv2.dnn.blobFromImage(이미지, scale factor, size, mean subtraction, swapRB, crop)
        #swapRB는 BRG로 인식되는 걸 RGB로 하라고 하는 것
        #crop 중앙부분을 자를까 말까 정해주는 것.
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers) #모델에 넣은것

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:

            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5: #detect하다 confidence가 0.5보다 크면 물체라 인식하겠다.
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)


        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = colors[i]

                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, color, 1)

        cv2.imshow("YOLOv3", frame)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break

    VideoSignal.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_YOLO_detection()
