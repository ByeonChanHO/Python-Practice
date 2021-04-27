from PIL import Image
import  numpy as np
import math
import time

#---------------------------------------------------------------------------
# MODULE : boxfilter
# INPUT  : boxNum - a box filter of size
# OUTPUT : a box filter array[boxNum, boxNum]
#----------------------------------------------------------------------------
def boxfilter(boxNum):

    #2D 필터에 행과 열의 수를 정하는 n이 홀수인지 체크합니다.
    assert(boxNum % 2 != 0), "Dimension must be odd"

    #n by n 에 0.04의 값으로 다 채워 만들어 리턴합니다.
    return np.full((boxNum, boxNum), 0.04)

#---------------------------------------------------------------------------
# MODULE : gauss1d
# INPUT  : sigma value
# OUTPUT : 1차원의 정규화 된 gaussian result value
#----------------------------------------------------------------------------
def gauss1d(sigma):

    #sigma 값이 반드시 0보다 커야하는 양수여야 한다.
    assert(sigma > 0), 'Sigma Value must be positive value'

    #필터의 길이를 sigma의 6배를 하고 소수점을 반올림한 수를 넣는다..
    FilterLenth = int(math.ceil(float(sigma) * 6))

    #필터의 길이가 짝수이면 +1해서 홀수로 만들어준다.
    if(FilterLenth%2 == 0) : FilterLenth = FilterLenth + 1

    #중앙이 몇 번째 배열인지 알아하기에 필터 배열길이을 2로 나눈다.
    Middle = int(FilterLenth/2)

    #x값이 중앙에서의 거리 차이만큼 값이 들어가도록 배열을 만든다.
    xValue = np.arange(-Middle, Middle + 1)

    #가우시안 함수를 만들어 주었다.
    Gaussian = lambda x : np.exp(-(x**2)/(2 * sigma**2))

    #가우시안의 공식에 x값을 넣어준어 결과를 받는다.
    GaussianResult = Gaussian(xValue)

    #배열을 정규화하여 합이 1이 되도록 한다.
    Result = GaussianResult / np.sum(GaussianResult)

    return Result

#---------------------------------------------------------------------------
# MODULE : gauss2d
# INPUT  : sigma value
# OUTPUT : Sigma value로 1차원에서 2차원으로 구한 Gussian Filter
#----------------------------------------------------------------------------
def gauss2d(sigma):

    #1D array를 함수gauss1d를 통해 받아온다.
    Gauss1D = gauss1d(sigma)

    #1D array에 외적(outer product)을 하여 2D array로 바꾸었다.
    Result = np.outer(Gauss1D, Gauss1D)

    total = np.sum(Result)  # 가우시안 필터의 합을 구하기 위한 변수
    Result /= total  # 정상화로 합이 1 이 되게 한다

    return Result


#---------------------------------------------------------------------------
# MODULE : convolve2d
# INPUT  : array - image, filter-kernel
# OUTPUT : Image와 filter로 합성곱 한 image
#----------------------------------------------------------------------------
def convolve2d(array, filter):

    #convolution하기 위해서는 일단 filter을 일단 위아래 좌우를 바꿔준다...
    filter = np.flipud(np.fliplr(filter))

    #커널과 이미지의 shape를 구한다.
    xKernelShape = filter.shape[0]  #Filter X
    yKernelShape = filter.shape[1]  #Filter Y
    xImgShape = array.shape[0]      #Array(Image) X
    yImgShape = array.shape[1]      #Array(Image) Y

    #출력에 대한 Array 선언 및 타입 설정
    Output = np.zeros((xImgShape,yImgShape))
    Output = Output.astype('float32')

    #zero padding된 공간 값을 찾는다 --> m = (f-1) / 2
    Padding = int((xKernelShape - 1)/2)

    #Padding 을 적용한다.
    if Padding != 0:
        ImagePadded = np.zeros((xImgShape + Padding*2, yImgShape + Padding*2))
        ImagePadded[Padding:-Padding, Padding:-Padding] = array
    else:
        ImagePadded = array #Padding이 없으면 원래 이미지의 array를 따른다.

    #합성공을 실행한다. 실질적으로 cross-correlation한다.
    for y in range(yImgShape):
        for x in range(xImgShape):
            Output[x,y] = (filter * ImagePadded[x:x+xKernelShape,y:y+yKernelShape]).sum()


    return Output

#---------------------------------------------------------------------------
# MODULE : gaussconvolve2d
# INPUT  : array - image, filter-kernel
# OUTPUT : Image와 Sigma로 gaussian filter와 convolve한 low filtering image
#----------------------------------------------------------------------------
def gaussconvolve2d(array,sigma):

    #필터로 가우시안 2D를 받아온다.
    Gauss2dFilter = gauss2d(sigma)

    #convolve2d에 적용한다.
    Result = convolve2d(array, Gauss2dFilter)

    return Result

#---------------------------------------------------------------------------
# MODULE : convolve2d_color
# INPUT  : array - image, filter-kernel
# OUTPUT : RGB array에 R/G/B 각각 gaussian filter 적용한 array
#----------------------------------------------------------------------------
def convolve2d_color(array,filter):
    
    # convolution하기 위해서는 일단 filter을 일단 위아래 좌우를 바꿔준다...
    filter = np.flipud(np.fliplr(filter))

    # 커널과 이미지의 shape를 구한다.
    xKernelShape = filter.shape[0]  # Filter X
    yKernelShape = filter.shape[1]  # Filter Y
    xImgShape = array.shape[0]  # Array(Image) X
    yImgShape = array.shape[1]  # Array(Image) Y

    # 출력에 대한 3D Array 선언 및 타입 설정
    Output = np.zeros((xImgShape, yImgShape,3))
    Output = Output.astype('float32')

    # zero padding된 공간 값을 찾는다 --> m = (f-1) / 2
    Padding = int((xKernelShape - 1) / 2)

    # Padding 을 적용한다.
    if Padding != 0:
        #3D RGB에서는 padding을 해줄 필요가 없기에 공간만 만들고 2D처럼 array를 배치한다..
        ImagePadded = np.zeros((xImgShape + Padding * 2, yImgShape + Padding * 2, 3))
        ImagePadded[Padding:-Padding, Padding:-Padding] = array
        #print(ImagePadded)
    else:
        ImagePadded = array  # Padding이 없으면 원래 이미지의 array를 따른다.

    # 합성공을 실행한다. 실질적으로 cross-correlation한다.
    for y in range(yImgShape):
        for x in range(xImgShape):
            Output[x, y, 0] = (filter * ImagePadded[x:x + xKernelShape, y:y + yKernelShape, 0]).sum()
            Output[x, y, 1] = (filter * ImagePadded[x:x + xKernelShape, y:y + yKernelShape, 1]).sum()
            Output[x, y, 2] = (filter * ImagePadded[x:x + xKernelShape, y:y + yKernelShape, 2]).sum()

    return Output

#---------------------------------------------------------------------------
# MODULE : gaussianconvolve2d_color
# INPUT  : array - image, filter
# OUTPUT : RGB blurred image
#----------------------------------------------------------------------------
def gaussianconvolve2d_color(array, sigma):
    # 주어진 시그마에 맞게 convolution 한다.
    return convolve2d_color(array, gauss2d(sigma))


#---------------------------------------------------------------------------
# MODULE : highfrequencyimage_get
# INPUT  : array - image, sigma
# OUTPUT : 고주파 이미지
#----------------------------------------------------------------------------
def highfrequencyimage_get(array,sigma):
    # image low frequency에서 128을 더하여 고주파 이미지로 변경
    return (array - gaussianconvolve2d_color(array,sigma) + 128)

#-----------------------------------------------------------------------------
print('Part 1 : Gaussian Filtering\n')
print('1. Show the results of your boxfilter(n) function for the cases n=3, n=4, n=7.')
print(boxfilter(3))
print(boxfilter(7))
time.sleep(0.1) #순서대로 결과값을 나오게 하기 위해 잠깐 멈추기
print(boxfilter(4))

print('2. Show the filter value s produced for sigma values of 0.3, 0.5, 1, and 2.')
print(gauss1d(0.3))
print(gauss1d(0.5))
print(gauss1d(1))
print(gauss1d(2))

print('3. Show the 2D Gaussian filter for sigma values of 0.5 and 1.')
print(gauss2d(0.5))
print(gauss2d(1))

print('4. Show both the original and filtered images.')
#이미지를 가져온다.
image = Image.open('2b_dog.bmp')
#원본 이미지를 보여준다.
image.show()

#회색깔(Grayscale)로 변경한다.
image = image.convert('L')
#matrix로 변환시킨다.
imageArray = np.asarray(image)

#gaussconvolve2d를 이용하여 low filtering image를 구한다.
FilteredImage = gaussconvolve2d(imageArray, 3)
#uint8의 타입으로 변환 후 array로 부터 PIL파일로 바꾼다.
FilteredImage = Image.fromarray(FilteredImage.astype(np.uint8))
#변환된 이미지 저장
FilteredImage.save('2b_dog_filtered_by_gaussconvolve2d.bmp','bmp')
print('2b_dog_filtered_by_gaussconvolve2d.bmp로 저장')
FilteredImage.show()

#-------------------------------------------------------------------------------------------------------------------

print('\nPart 2: Hybrid Image\n')

print('1. Gaussian filtered low frequency image')
image = Image.open('1a_bicycle.bmp') #이미지를 불러온다.
imageArray = np.asarray(image)      #이미지 PIL를 배열로 변경
imageArray = imageArray.astype('float32') #타입 float32로 변경

LowFrequencyImage = gaussianconvolve2d_color(imageArray,3)  #low frequency Image 구하기
image_LowFrequncy = LowFrequencyImage
LowFrequencyImage = Image.fromarray(LowFrequencyImage.astype(np.uint8)) #PIL로 변경

LowFrequencyImage.save('low_frequency_image.bmp', 'bmp') #저장
print('low_frequency_image.bmp을 저장 완료')
LowFrequencyImage.show()

print('2. Gaussian filtered high frequency image')
# Low filtering Image할 그림 파일을 열어 array로 받는다
image = Image.open('1b_motorcycle.bmp')
# image로 부터 image array put
imageArray = np.asarray(image)
imageArray = imageArray.astype('float32') # array type = float32

# High frequency image 를 위해 함수를 콜하고 uint8로 형변환 후 image로 변환
HighFrequencyImage = highfrequencyimage_get(imageArray, 3)
image_highfrequency = HighFrequencyImage # hybid image에 사용하기 위해 임시저장한다
HighFrequencyImage = Image.fromarray(HighFrequencyImage.astype(np.uint8)) #PIL로 변경
# Filtered 이미지 저장한다.
HighFrequencyImage.save('high_frequency_image.bmp', 'bmp')
HighFrequencyImage.show()
print('high_frequency_image.bmp을 저장 완료하였습니다.')



print('3. Gaussian hybrid image')
image_hybrid = image_LowFrequncy + image_highfrequency - 128 # 128 없이 이미지 추출
# RGB 값이 0 ~ 255사이에 있으므로 유효하도록 하기 위해 다음과 같이 한다.
image_hybrid[image_hybrid > 255] = 255       # 255 보다 높은 주파수이면 Max 255
image_hybrid[image_hybrid < 0] = 0           # 0 보다 낮은 주파수이면 Min 0
# 이미지화 하기 위해 다시 uint8로 형변환 후 image로 변환
image_hybrid = Image.fromarray(image_hybrid.astype(np.uint8))
image_hybrid.save('hybrid_image.bmp', 'bmp')
image_hybrid.show()
print('hybrid_image.bmp을 저장 완료하였습니다.')
