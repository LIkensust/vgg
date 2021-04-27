import cv2
def OpenImage(path, reshape=True, size=224):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if reshape:
        img = cv2.resize(img,(size,size))
    return img