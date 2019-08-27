import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread("D:\\lena.png")

conv = np.zeros([5, 5]) # 高斯卷积核
sigma1 = 1 # 方差
sigma2 = sigma1 * sigma1
sum = 0

for i in range(5):
    for j in range(5):
        conv[i, j] = math.exp((-(i - 3) * (i - 3) - (j - 3) *  (j - 3)) / (2 * sigma2)) / (2 * math.pi * sigma2)
        sum = sum + conv[i, j]

gaussian = conv / sum # 归一化

# 灰度化
# 计算公式为：0.299 * R(i, j) + 0.587 * G(i, j) + 0.114 * B(i, j)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR格式转换为RGB格式
img_gray = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])

W, H = img_gray.shape
new_gray = np.zeros([W - 5, H - 5])
for i in range(W - 5):
    for j in range(H - 5):
        new_gray[i, j] = np.sum(img_gray[i:i+5,j:j+5] * gaussian)

# 计算梯度幅值
W_new, H_new = new_gray.shape
dx = np.zeros([W_new - 1, H_new - 1])
dy = np.zeros([W_new - 1, H_new - 1])
M = np.zeros([W_new - 1, H_new - 1])
theta = np.zeros([W_new - 1, H_new - 1])

for i in range(W_new - 1):
    for j in range(H_new - 1):
        dx[i, j] = new_gray[i + 1, j] - new_gray[i, j]
        dy[i, j] = new_gray[i, j + 1] - new_gray[i, j]
        M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
        #theta = math.atan(dx[i, j] / dy[i, j] + 0.00000000000001)

# 非极大值抑制
W_M, H_M = M.shape
NMS = np.copy(M)
NMS[0, :] = NMS[W_M - 1, :] = NMS[:, 0] = NMS[:, H_M - 1] = 0
for i in range(1, W_M - 1):
    for j in range(1, H_M - 1):
        # 当前梯度为0，则不是边缘点
        if M[i, j] == 0:
            NMS[i, j] = 0
        else:
            gradX = dx[i, j]
            gradY = dy[i, j]
            gradTemp = M[i, j]

            if np.abs(gradX) > np.abs(gradY):
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = M[i, j - 1]
                grad4 = M[i, j + 1]
                if gradX * gradY > 0:
                    grad1 = M[i + 1, j - 1]
                    grad3 = M[i - 1, j + 1]
                else:
                    grad1 = M[i - 1, j - 1]
                    grad3 = M[i + 1, j + 1]
            
            else:
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = M[i - 1, j]
                grad4 = M[i + 1, j]
                if gradX * gradY > 0:
                    grad1 = M[i - 1, j - 1]
                    grad3 = M[i + 1, j + 1]
                else:
                    grad1 = M[i - 1, j + 1]
                    grad3 = M[i + 1, j - 1]

            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[i, j] = gradTemp
            else:
                NMS[i, j] = 0

# 双阈值算法检测连接边缘
W3, H3 = NMS.shape
DT = np.zeros([W3, H3])               
# 定义高低阈值
TL = 0.2 * np.max(NMS)
TH = 0.3 * np.max(NMS)
for i in range(1, W3-1):
    for j in range(1, H3-1):
        if (NMS[i, j] < TL):
            DT[i, j] = 0
        elif (NMS[i, j] > TH):
            DT[i, j] = 1
        elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
              or (NMS[i, [j-1, j+1]] < TH).any()):
            DT[i, j] = 1
        
plt.imshow(DT, cmap = "gray")
plt.show()