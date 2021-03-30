# CycleGan

###	내용 요약
CycleGAN 은 pix2pix에서 GAN을 거처 짝이 없는(unpair) 이미지도 변형을 할 수 있게 만든 발전된 기법이다. 
CycleGAN은 Cycle- Consistent Adversarial Nets을 이용하여 스스로 이미지에 대한 변형 및 평가, 피드백을 하고 스스로 이미지 변형에 학습하는 원리를 가지고 있다. 
하지만 이미지의 잘못된 위치 변형을 뜻하는 Reconstruction error가 자주 발생하면서 이를 방지하기 위해 변형된 사진을 다시 한번 원본 사진으로 변형하여 원래 원본 사진과 비교하는 Cycle Consistency Loss를 도입했다. 
그 이후 Reconstruction error에 대해서도 피드백을 받아 학습을 할 수 있게 되어 Unpair한 이미지도 변형이 가능하도록 발전되었다.


###	개선 방안
CycleGAN에서 배치 사이즈(Batch size) 값을 1보다 더 큰 값으로 넣으면 입력(input)에 들어가는 이미지가 한번에 10개의 묶음의 그룹으로 들어가여 학습이 진행 되기에 학습속도를 높일 수 있습니다. 또한 epoch의 값을 40이 아닌 절반의 값 20으로 설정하면 학습 횟수가 줄어들어 더욱더 속도를 높일 수 있습니다.
CycleGAN 에 gernerator을 unet이 아닌 레이어(layer), 한도(parameter)가 더 큰 resent로 바꾸어 정규화(normalization)을 instance가 아닌 각 레이어(layer)마다 정규화(normalization)하는 레이어(layer)를 두어, 변형된 분포가 나오지 않도록 조절하는 batch로 정확도를 더욱 올릴 수 있다.


업로드 정리중..


#### 참조
https://www.tensorflow.org/tutorials/generative/cyclegan
