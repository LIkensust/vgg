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

logger = logging.getLogger('main_log')

class ImageMerger:
    def __init__(self, vggModelPath='module/vgg16_weights.npz', modelPath=None, load = True):
        self._fg = feature_generator.FeatureGenerator(vggModelPath)
        if (not modelPath is None) and (load is True):
            self._model = tf.keras.models.load_model(modelPath)
        else:
            self._model = None
        self._modelPath = modelPath

    def makeSample(self, imgPath, sampleSize=100, rate=0.3):
        self._samples = []
        paths = os.listdir(imgPath)
        imgs = []
        for p in paths:
            if p.find('jp') == -1:
                continue
            print(imgPath + p)
            imgs.append(openimg.OpenImage(imgPath + p, False))
        for i in tqdm.tqdm(range(sampleSize)):
            index = random.randint(0,len(imgs)-1)
            img = imgs[index]
            h, w, _ = img.shape
            ul = [300 + random.randint(-100,100),300 + random.randint(-100,100)]
            o, n, H, HL, HV = perspective_param_generator.GenerateRandomPerspectiveTransform(ul, 224, rate)
            img_t = cv2.warpPerspective(img, HV, dsize=(w, h))
            piece = img[o[0][0]:o[2][0],o[0][1]:o[1][1]]
            piece_t = img_t[o[0][0]:o[2][0],o[0][1]:o[1][1]]
            # Trans H
            tx = o[0][0]
            ty = o[0][1]
            for i in range(4):
                o[i][0] -= tx
                n[i][0] -= tx
                o[i][1] -= ty
                n[i][1] -= ty
            _, HL = perspective_param_generator.CalculatePerspectiveTransform(o, n)
            baseFeature = self._fg.MakeFeature(piece)
            subFeature = self._fg.MakeFeature(piece_t)
            feature = self._fg.MakeDiffFeature(baseFeature, subFeature)
            self._samples.append([feature, HL])

    def Train(self, imgPath, sampleSize=100, times=20, batch=20):
        self.makeSample(imgPath,sampleSize)
        x = np.array([self._samples[i][0] for i in range(sampleSize)])
        y = np.array([self._samples[i][1] for i in range(sampleSize)])

        logger.debug("SAMPLES X SAPHE : [{}]".format(x.shape))
        logger.debug("SAMPLES Y SAPHE : [{}]".format(y.shape))



        self._model = tf.keras.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1, 144)),
            tf.keras.layers.Dense(1000, activation='relu', input_shape=x.shape[1:]),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(8)
        ])

        exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=10, decay_rate=0.99)

        self._model.compile(optimizer=tf.keras.optimizers.SGD(exponential_decay),
                      loss='mean_squared_error',
                      # loss='MSE',
                      metrics=['accuracy'])

        self._model.fit(x, y, epochs=times, batch_size=batch)

        if self._modelPath is None:
            self._modelPath = 'module/image_merger'
        self._model.save(self._modelPath)

    def Predict(self, img1, img2):
        baseFeature = self._fg.MakeFeature(img1)
        feature = self._fg.MakeFeature(img2)
        subFeature = self._fg.MakeDiffFeature(baseFeature, feature)
        x = np.array([subFeature])
        py = self._model.predict(x)[0]
        py = np.append(py,1).resize((3,3))
        return py


