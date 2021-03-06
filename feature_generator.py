import vgg16 as vgg
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
from hilbert_index import HILBERT_INDEX_14M14

tf.disable_eager_execution()
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger('main_log')

class FeatureGenerator:
    def __init__(self,model):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = vgg.vgg16(self.imgs, model, self.sess)

    # 输入图像的大小应该是 244*244
    def MakeFeature(self, img):
        #layers = [self.vgg.pool3, self.vgg.pool4, self.vgg.pool5]
        layers = [self.vgg.pool4, self.vgg.pool5]
        results = self.sess.run(layers, feed_dict={self.vgg.imgs: [img]})
        # 所有的输出都带有多出的一个维度 因为输入是支持多图的 先消除这个维度
        # 将所有特征标准化
        for i in range(len(results)):
            results[i] = results[i][0]
            #std = np.std(results[i])
            results[i] /= 1000

        results[1] = np.kron(results[1],np.array([[[1],[1]],[[1],[1]]]))
        #results[2] = np.kron(results[2],np.array([[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]))
        return results

    def Distance(self, v1, v2):
        return np.linalg.norm(v1-v2)

    '''
    def MakeDiffFeature(self, f1,f2):
        ret = np.zeros((14*14,14*14))
        for i in range(14*14):
            for j in range(14*14):
                tmp = (self.Distance(f1[0][int(i / 14)][int(i % 14)], f2[0][int(j / 14)][int(j % 14)]) + \
                            self.Distance(f1[1][int(i / 14)][int(i % 14)], f2[1][int(j / 14)][int(j % 14)]))
                ret[i][j] = (0.5 - abs(1.0/(1 + np.exp(tmp)) - 0.5))*2
        ret = np.resize(ret,(14*14*14*14))
        return ret
    '''

    def MakeDiffFeature(self, f1,f2):
        ret = np.zeros((16 * 16, 16 * 16))
        for i in range(14 * 14):
            for j in range(14 * 14):
                tmp = (self.Distance(f1[0][int(i / 14)][int(i % 14)], f2[0][int(j / 14)][int(j % 14)]) + \
                       self.Distance(f1[1][int(i / 14)][int(i % 14)], f2[1][int(j / 14)][int(j % 14)]))
                ret[i][HILBERT_INDEX_14M14[j]] = (0.5 - abs(1.0 / (1 + np.exp(tmp)) - 0.5)) * 2
        ret = np.resize(ret, (16 * 16 * 16 * 16))
        return ret


