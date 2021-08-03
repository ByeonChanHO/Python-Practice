import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, convolve
import scipy

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    A = np.zeros((n,9))

    # build matrix for equations in Page 52
    # Af = 0 의 A 메트릭스를 구한다.
    for i in range(n) :
        A[i] = [x1[0,i] * x2[0,i], x1[0,i] * x2[1,i], x1[0,i] * x2[2,i],
                x1[1,i] * x2[0,i], x1[1,i] * x2[1,i], x1[1,i] * x2[2,i],
                x1[2,i] * x2[0,i], x1[2,i] * x2[1,i], x1[2,i] * x2[2,i]]

    # compute the solution in Page 52
    #least square solution을 구한다.
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F: make rank 2 by zeroing out last singular value (Page 53)
    #rank 2 matrix로 만들기 위해 S[2]에 0을 넣어 last singular value를 구한다.
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    S = np.diag(S)

    #F = USV 를 계산한다.
    SV = np.matmul(S,V)
    F = np.matmul(U, SV)

    ### YOUR CODE ENDS HERE
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1

    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    #e1(epipole1) 을 구한다.
    U, S, V = np.linalg.svd(F)
    e1 = V[-1] / V[-1][-1] #normalize

    #e2(epipole2)을 구한다.
    U, S, V = np.linalg.svd(F.T)
    e2 = V[-1] / V[-1][-1] #normalize
    ### YOUR CODE ENDS HERE
    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE

    #화면 분할한 첫번째에 img1 보여준다.
    plt.subplot(1, 2, 1, aspect='equal')
    plt.imshow(img1)

    #이미지 특징 개수 len 과 방정식 기울기와 상수 초기화
    len = cor1.shape[1]
    gradient = np.zeros(len,dtype=float)
    constant = np.zeros(len,dtype=float)

    #image1의 특징을 scatter로 표시하고
    # 카메라와 각 특징들의각 방정식의 기울기 상수를 구한다.
    for i in range(len) :
        plt.scatter(x = cor1[0][i], y = cor1[1][i],s = 15, c='r')
        gradient[i] = (cor1[1][i] - e2[1])/(cor1[0][i] - e2[0])
        constant[i] = cor1[1][i] -(cor1[0][i] * gradient[i])

    #구한 기울기와 상수로 이미지에 선을 그린다.
    for i in range(len) :
        x_value = np.arange(0, max(img1.shape[0], img1.shape[1]))
        y_value = (x_value * gradient[i]) + constant[i]
        valid_index = ((y_value >= 0) & (y_value <= min(img1.shape[0] ,img1.shape[1])-1))
        plt.plot(x_value[valid_index], y_value[valid_index])

    #화면 분할한 두번째 이미지2
    plt.subplot(1, 2, 2, aspect='equal')
    plt.imshow(img2)

    # image2의 특징을 scatter로 표시하고
    # 카메라와 각 특징들의각 방정식의 기울기 상수를 구한다.
    for i in range(len) :
        plt.scatter(x = cor2[0][i], y = cor2[1][i],s = 15, c='r')
        gradient[i] = (cor2[1][i] - e1[1])/(cor2[0][i] - e1[0])
        constant[i] = cor2[1][i] -(cor2[0][i] * gradient[i])

    # 구한 기울기와 상수로 이미지에 선을 그린다.
    for i in range(len) :
        x_value = np.arange(0, max(img1.shape[0], img1.shape[1]))
        y_value = (x_value * gradient[i]) + constant[i]
        valid_index = ((y_value >= 0) & (y_value <= max(img1.shape[0], img1.shape[1])))
        plt.plot(x_value[valid_index], y_value[valid_index])

    plt.show()
    ### YOUR CODE ENDS HERE

    return

#워리어 실행
draw_epipolar_lines(img1, img2, cor1, cor2)

#graffiti_a 와 graffiti_b 실행
img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')
cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")
draw_epipolar_lines(img1, img2, cor1, cor2)