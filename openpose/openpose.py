# References
# https://github.com/quanhua92/human-pose-estimation-opencv

#2D 포즈로 볼수 있는데 CNN를 돌려서 각 부위가 있을 곳을 색칠을 한다.
#Part Confidence Maps, Part Affinity Fields
#bipartite Matching 미분 매칭이 화살표가 있는데 인체 붙여지는 걸 하는 것이다. Part Affinity Fields 로
#확률을 보고 선을 그어준다 그래서 Parsing Results가 나온다.
#간단하게 요약하면 신체 부위가 있을 것 같은 정도에 인코딩
# 신체 부위~

#Mobile Net 기반 신경망
#openPose 가 경량 모델로 구현 되었다.
#경량 모델은 ?
# 클라우드 컴퓨팅 : ~
# 엣지 컴퓨팅 : ~

#작은 신경망 만들기
#작다라는 기준 시간이 적게 걸리는 걸 작은 신경망이라 한다.
#밑에 껏들 다 조사해보자
#Kernel Reduction 은 성능을 생각해서 적당히 줄여야한다.
#Channel Reduction.???

#Body part
#   몸부분과 연결하는 쌍을 구성할 것이다.
#   각 채널이 어ㄸㅎ게 하는 지 모른다.
# Nose 라는 0채널 등등

#part prediction
#   CNN 통해서 어떤 part인ㄴ지 감지한다.
#   그걸 input 넣으면 됨
#   하나의 사진에 한명만 하겠다 local Maximum 대신 하나의 global Maximum 을 하겠다

#예시를 주는 것이 좋다 리포트에

#CNN이 뭐냐 여기서 OPenpos 와 Yolo를 공부해봣다
# 리포트에 넣었을 것은 어떤걸 해봐서 어디에 필요할 것이다.라고 써봐라

import cv2
import numpy as np
import pafy


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

url = "https://www.youtube.com/watch?v=Z9Sn9r82gyE"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

VideoSignal = cv2.VideoCapture()
VideoSignal.open(best.url)

cnt = 0

while cv2.waitKey(1) < 0:
    hasFrame, frame = VideoSignal.read()

    # 성능을 위해 프레임 생략
    cnt += 1
    if cnt % 4 != 0:
        continue

    if not hasFrame:
        cv2.waitKey()
        break

    frame = cv2.resize(frame,(368,368))
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > 0.2 else None)
        #0.2 보다 높아야 파트로 받아들겠다. conf = confidence
    

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        #연결할게 있으면 연결 없으면 스킵
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('OpenPose using OpenCV Go', frame)