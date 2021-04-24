import feature_generator
import logging
import cv2
import image_merger
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][line:%(lineno)d][%(levelname)s]:[%(message)s]')

logger = logging.getLogger('main_log')

def OpenImage(path):
    img = cv2.imread('images/apple.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.debug("IMG : shape [{}]".format(img.shape))
    img = cv2.resize(img,(224,224))
    return img

def test(path):
    #fg = feature_generator.FeatureGenerator("module/vgg16_weights.npz")
    logger.debug("create feature generator success")
    img = OpenImage("images/apple.jpg")
    mg = image_merger.ImageMerger("module/vgg16_weights.npz", path, load=False)
    mg.Train(img,200,400)
    #mg.Predict(img,img)

def showImage(img):
    cv2.imshow("image", img)
    cv2.waitKey()

def warpImage(img1,img2,show=False):
    h,w = img1.shape
    img = np.zeros((h,w*2),np.uint8)
    img[0:h, 0:w] = img1
    img[0:h, w:2*w] = img2
    if show:
        showImage(img)
    return img

def test1(path):
    mg = image_merger.ImageMerger("module/vgg16_weights.npz", path, load=True)
    h = w = 224
    img = OpenImage("images/apple.jpg")
    for i in range(100):
        H, sita = mg.RandHSmall(w, h, 0.3)
        imgH = cv2.warpPerspective(img, H, dsize=(w, h))
        baseFeature = mg._fg.MakeFeature(img)
        feature = mg._fg.MakeFeature(imgH)
        subFeature = mg._fg.MakeDiffFeature(baseFeature, feature)
        x = np.array([subFeature])
        py = mg._model.predict(x)
        py = float(py[0])

        nH = np.array([
            [np.cos(py),-1.0*np.sin(py),0],
            [np.sin(py),np.cos(py),0],
            [0.0,0.0,1.0],
        ])
        imgHH = cv2.warpPerspective(img,nH,dsize=(w, h))
        imghstack = np.hstack((imgH, imgHH))
        showImage(imghstack)
        #showImage(imgHH)
        #warpImage(imgH, imgHH, True)


if __name__ == "__main__":
    path = "module/train_1_200_400_1_03"
    logger.debug("OPENCV VERSION [{}]".format(cv2.__version__))
    #test(path)
    test1(path)