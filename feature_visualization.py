import feature_generator
import random
import cv2
import numpy as np
import tensorflow as tf
import tqdm
import logging
import perspective_param_generator
import os
import openimg

fg = feature_generator.FeatureGenerator("module/vgg16_weights.npz")

img1 = openimg.OpenImage("./images/1.jpeg",False)
img2 = openimg.OpenImage("./images/2.jpeg",False)
img3 = openimg.OpenImage("./images/3.jpeg",False)

w1,h1,_ = img1.shape
w2,h2,_ = img2.shape
w3,h3,_ = img3.shape

o, n, H, HL, HV = perspective_param_generator.GenerateRandomPerspectiveTransform([100,100], 224, 0.3)
img_t = cv2.warpPerspective(img1, HV, dsize=(h1, w1))
piece = img1[o[0][0]:o[2][0], o[0][1]:o[1][1]]
piece_t = img_t[o[0][0]:o[2][0], o[0][1]:o[1][1]]

print(piece.shape)
baseFeature = fg.MakeFeature(piece)
subFeature = fg.MakeFeature(piece_t)
feature = fg.MakeDiffFeature(baseFeature, subFeature)
feature1 = np.resize(feature,(16*16,16*16))


img_t = cv2.warpPerspective(img2, HV, dsize=(h2, w2))
piece = img2[o[0][0]:o[2][0], o[0][1]:o[1][1]]
piece_t = img_t[o[0][0]:o[2][0], o[0][1]:o[1][1]]
baseFeature = fg.MakeFeature(piece)
subFeature = fg.MakeFeature(piece_t)
feature = fg.MakeDiffFeature(baseFeature, subFeature)
feature2 = np.resize(feature,(16*16,16*16))

fs1 = np.vstack((feature1,feature2))

o, n, H, HL, HV = perspective_param_generator.GenerateRandomPerspectiveTransform([100,100], 224, 0.6)
img_t = cv2.warpPerspective(img1, HV, dsize=(h1, w1))
piece = img1[o[0][0]:o[2][0], o[0][1]:o[1][1]]
piece_t = img_t[o[0][0]:o[2][0], o[0][1]:o[1][1]]

print(piece.shape)
baseFeature = fg.MakeFeature(piece)
subFeature = fg.MakeFeature(piece_t)
feature = fg.MakeDiffFeature(baseFeature, subFeature)
feature1 = np.resize(feature,(16*16,16*16))


img_t = cv2.warpPerspective(img2, HV, dsize=(h2, w2))
piece = img2[o[0][0]:o[2][0], o[0][1]:o[1][1]]
piece_t = img_t[o[0][0]:o[2][0], o[0][1]:o[1][1]]
baseFeature = fg.MakeFeature(piece)
subFeature = fg.MakeFeature(piece_t)
feature = fg.MakeDiffFeature(baseFeature, subFeature)
feature2 = np.resize(feature,(16*16,16*16))

fs2 = np.vstack((feature1,feature2))

fs = np.hstack((fs1,fs2))

cv2.imshow("features", fs)
cv2.waitKey()