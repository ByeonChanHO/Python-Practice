Yolo 프로그램 연습

Yolo 결과
![week2-1](https://user-images.githubusercontent.com/38696775/111876009-05ec8b80-89e0-11eb-81ac-880da07de0aa.jpg)


Yolo 의 장점
1. 간단한 처리과정으로 매우 빠르다.
2. background error가 낮다
3. object에 대한 더 일반화된 특징을 학습한다.

동작 방법
  이미지를 448 X 448로 변경 후 S X S grid로 나눈다.
  각 cell에 대한 C개의 클래스에 대한 예측값 생성, B개의 bounding box 예측
  Class에 대한 예측은 경우 색으로 구분
  Bounding box는 confidence와 합해져 구분
  위 결과를 종합해 detection map을 생성한다. (물체를 디텍션한다.)

YOLO 프레임워크
  Darknet : C언어로 작성된 물체 인식 오픈 소스 신경망
  DarkFlow : Tensorflow 를 활용한 YOLO 신경망
  OpenCV - 현재 여기에 올린 코드

