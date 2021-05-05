import cv2
import numpy as np
import matplotlib.pyplot as plot


def GetRowKernal(size, wide, unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(size):
        if( (int(i / wide) % 2) == 1):
            ret[i,:] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetColKernal(size, wide, unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(size):
        if( (int(i / wide) % 2) == 1):
            ret[:,i] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetUL2RDKernal(size, wide,unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(size):
        for j in range(size):
            m = max(i,j)
            if int(m**2/(2*wide)) % 2 == 1:
                ret[i][j] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetUL2RDLineKernal(size, wide,unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(size):
        for j in range(size):
            m = i+j
            if int(m**2/(2*wide)) % 2 == 1:
                ret[i][j] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetConcentricCircleKernal(size, wide, unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    mid = size / 2
    for i in range(size):
        for j in range(size):
            if int((((i-mid)**2 + (j-mid)**2)**0.5) / wide) % 2 == 1:
                #print(i,j)
                ret[i][j] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetCrossKernal(size, wide, unflod=True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(int(size/2)):
        for j in range(int(size/2)):
            m = max(i, j)
            if int(m ** 2 / (2 * wide)) % 2 == 1:
                ret[i][j] = 1.0
    ret[:int(size/2),int(size/2):] = np.fliplr(ret[:int(size/2),:int(size/2)])
    ret[int(size / 2):,:] = np.flipud(ret[:int(size/2),:])
    if unflod:
        return np.resize(ret, (size * size))
    return ret

def GetCrossLineKernal(size, wide, unflod=True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(int(size/2)):
        for j in range(int(size/2)):
            m = i+j
            if int(m ** 2 / (2 * wide)) % 2 == 1:
                ret[i][j] = 1.0
    ret[:int(size/2),int(size/2):] = np.fliplr(ret[:int(size/2),:int(size/2)])
    ret[int(size / 2):,:] = np.flipud(ret[:int(size/2),:])
    if unflod:
        return np.resize(ret, (size * size))
    return ret

def GetSinKernal(size, wide, unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    times = int(size / wide)
    for i in range(size):
        y = int(np.sin(i * 2*np.pi / wide) * wide)
        for j in range(wide):
            y += 1
            for k in range(times):
                ny = y+wide*k*2
                if ny < size and ny > 0:
                    ret[i][ny] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return ret

def GetDotKernal(size, wide, unflod = True):
    ret = np.zeros((size, size)).astype(np.float_)
    for i in range(size):
        for j in range(size):
            if i % wide == 1 and j % wide == 1:
                ret[i][j] = 1.0
    if unflod:
        return np.resize(ret, (size*size))
    return



if __name__ == "__main__":
    cir = GetCrossLineKernal(300, 3, False)
    plot.imshow(cir)
    plot.show()