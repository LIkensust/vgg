import feature_generator
import random
import cv2
import numpy as np
import tensorflow as tf
import tqdm
import logging

logger = logging.getLogger('main_log')

class ImageMerger:
    def __init__(self, vggModelPath='module/vgg16_weights.npz', modelPath=None, load = True):
        self._fg = feature_generator.FeatureGenerator(vggModelPath)
        if (not modelPath is None) and (load is True):
            self._model = tf.keras.models.load_model(modelPath)
        else:
            self._model = None
        self._modelPath = modelPath

    def RandHSmall(self, w, h, rate=0.5):
        H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        sita = random.uniform(-1.0 * rate, 1 * rate)
        H[0][0] = np.cos(sita)
        H[0][1] = -1 * np.sin(sita)
        H[1][0] = np.sin(sita)
        H[1][1] = np.cos(sita)
        return np.array(H), sita

    def tranH(self, H):
        HL = []
        for i in range(3):
            for j in range(3):
                HL.append(H[i][j])
        return HL[:-1]

    def Train(self, img, sampleSize=100, times=1000, batch=20):
        self._samples = []
        #img = cv2.resize(img,(224,224))
        h = w = 224
        baseFeature = self._fg.MakeFeature(img)
        for i in tqdm.tqdm(range(sampleSize)):
            H,sita = self.RandHSmall(w,h,0.3)
            imgH = cv2.warpPerspective(img, H, dsize=(w, h))
            feature = self._fg.MakeFeature(imgH)
            subFeature = self._fg.MakeDiffFeature(baseFeature, feature)
            self._samples.append([subFeature, [sita]])

        x = np.array([self._samples[i][0] for i in range(sampleSize)])
        y = np.array([self._samples[i][1] for i in range(sampleSize)])

        logger.debug("SAMPLES X SAPHE : [{}]".format(x.shape))
        logger.debug("SAMPLES Y SAPHE : [{}]".format(y.shape))

        self._model = tf.keras.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1, 144)),
            tf.keras.layers.Dense(1000, activation='relu', input_shape=x.shape[1:]),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self._model.compile(optimizer=tf.keras.optimizers.SGD(0.0001),
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
        return py


