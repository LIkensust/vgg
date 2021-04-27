import cv2
import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
from openimg import OpenImage
from warp_two_image import warpTwoImages

'''  Copy from OpenCV::getPerspectiveTransform  
    | x |   | c00 c01 c02 |   | x |   | x' |
H * | y | = | c10 c11 c12 | * | y | = | y' |
    | 1 |   | c20 c21 c22 |   | 1 |   | w  |

u = x' / w
v = y' / w   
    
/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   cij - matrix coefficients, c22 = 1
 */
'''
def CalculatePerspectiveTransform(ps1,ps2):
    '''
    Input :
        ps1 is [[x0,y0],...,[x3,y3]]
        ps2 is [[u0,v0],...,[u3,v3]]
    Output:
        H is 3*3 matrix
        HL is 1*8 array
    '''
    x0, y0 = ps1[0][0], ps1[0][1]
    x1, y1 = ps1[1][0], ps1[1][1]
    x2, y2 = ps1[2][0], ps1[2][1]
    x3, y3 = ps1[3][0], ps1[3][1]

    u0, v0 = ps2[0][0], ps2[0][1]
    u1, v1 = ps2[1][0], ps2[1][1]
    u2, v2 = ps2[2][0], ps2[2][1]
    u3, v3 = ps2[3][0], ps2[3][1]

    A = np.array([
        [x0, y0,  1,  0,  0,  0, -1.0*x0*u0, -1.0*y0*u0],
        [x1, y1,  1,  0,  0,  0, -1.0*x1*u1, -1.0*y1*u1],
        [x2, y2,  1,  0,  0,  0, -1.0*x2*u2, -1.0*y2*u2],
        [x3, y3,  1,  0,  0,  0, -1.0*x3*u3, -1.0*y3*u3],
        [ 0,  0,  0, x0, y0,  1, -1.0*x0*v0, -1.0*y0*v0],
        [ 0,  0,  0, x1, y1,  1, -1.0*x1*v1, -1.0*y1*v1],
        [ 0,  0,  0, x2, y2,  1, -1.0*x2*v2, -1.0*y2*v2],
        [ 0,  0,  0, x3, y3,  1, -1.0*x3*v3, -1.0*y3*v3],
    ])

    B = np.array([
        [u0],
        [u1],
        [u2],
        [u3],
        [v0],
        [v1],
        [v2],
        [v3],
    ])
    HL = np.matmul(np.linalg.inv(A),B)
    H = HL.copy()
    H.resize((3,3))
    H[2][2] = 1
    return H, HL.reshape(8)


def GenerateRandomPerspectiveTransform(ps, length, rate):
    '''
    Input :
        ps is a upper left corner of the square : [x,y]  Note : in opencv img, upper left is (0,0) x to right, y to down
        length is the length of the square
        rate is random rate, random range will be : [-length * rate * 0.5,length * rate * 0.5]
    Output :
        src square format as [[x_ul,y_ul],[x_dl,y_dl],[x_ur,y_ur],[x_dr,y_dr]]
        transd square format as [[x_ul,y_ul],[x_dl,y_dl],[x_ur,y_ur],[x_dr,y_dr]]
        PerspectiveTransform Matix H 3*3
        HL transed from H, 1*8
        HV inv of H
    '''
    ul = [ps[0], ps[1]]
    dl = [ps[0], ps[1] + length]
    ur = [ps[0] + length, ps[1]]
    dr = [ps[0] + length, ps[1] + length]

    # random range
    rr = [-1.0*length * rate * 0.5,length * rate * 0.5]

    # random points
    ul_n = [ul[0] + random.uniform(rr[0],rr[1]), ul[1] + random.uniform(rr[0],rr[1])]
    dl_n = [dl[0] + random.uniform(rr[0],rr[1]), dl[1] + random.uniform(rr[0],rr[1])]
    ur_n = [ur[0] + random.uniform(rr[0],rr[1]), ur[1] + random.uniform(rr[0],rr[1])]
    dr_n = [dr[0] + random.uniform(rr[0],rr[1]), dr[1] + random.uniform(rr[0],rr[1])]

    # plt.plot(ul[0],-1.0*ul[1], 'ro')
    # plt.plot(dl[0],-1.0*dl[1], 'ro')
    # plt.plot(ur[0],-1.0*ur[1], 'ro')
    # plt.plot(dr[0],-1.0*dr[1], 'ro')
    # plt.plot(ul_n[0],-1.0*ul_n[1], 'bo')
    # plt.plot(dl_n[0],-1.0*dl_n[1], 'bo')
    # plt.plot(ur_n[0],-1.0*ur_n[1], 'bo')
    # plt.plot(dr_n[0],-1.0*dr_n[1], 'bo')
    # plt.show()

    old_points = [ul,dl,ur,dr]
    new_points = [ul_n,dl_n,ur_n,dr_n]

    # calculate perspective transform H
    H, HL = CalculatePerspectiveTransform(old_points, new_points)
    HV = np.linalg.inv(H)

    return old_points, new_points, H, HL, HV




def testCalculatePerspectiveTransform():
    ps = [[0.0,0.0], [1.0,0.0], [0.1,3.0], [1.5,2.5]]
    ps_1 = []
    for p in ps:
        ps_1.append(np.array([p[0],p[1], 1]))
    ps_1 = np.array(ps_1).T
    H = np.array([
        [0.5,  1.2, 12],
        [2  , -0.2, 4 ],
        [5  , -4  , 1 ]
    ])
    ps_2 = numpy.matmul(H, ps_1).T
    ps2 = []
    for p in ps_2:
        ps2.append([p[0] / p[2], p[1] / p[2]])
    H,HL = CalculatePerspectiveTransform(ps,ps2)
    print(H, HL)

def testGenerateRandomPerspectiveTransform():
    img = OpenImage('images/square.jpeg', False)
    h,w,_ = img.shape

    size = 224
    o,n,H,HL,HV = GenerateRandomPerspectiveTransform([100,100], size, 0.3)
    n = np.array(n).astype(np.int)

    # cv2.line(img, pt1=(o[0][0],o[0][1]), pt2=(o[1][0],o[1][1]), color=(255,100,0), thickness=5)
    # cv2.line(img, pt1=(o[2][0],o[2][1]), pt2=(o[3][0],o[3][1]), color=(255,100,0), thickness=5)
    # cv2.line(img, pt1=(o[0][0],o[0][1]), pt2=(o[2][0],o[2][1]), color=(255,100,0), thickness=5)
    # cv2.line(img, pt1=(o[3][0],o[3][1]), pt2=(o[1][0],o[1][1]), color=(255,100,0), thickness=5)
    #
    # cv2.line(img, pt1=(o[0][0]+100,o[0][1]+100), pt2=(o[1][0]+100,o[1][1]+100), color=(0,255,0), thickness=5)
    # cv2.line(img, pt1=(o[2][0]+100,o[2][1]+100), pt2=(o[3][0]+100,o[3][1]+100), color=(0,255,0), thickness=5)
    # cv2.line(img, pt1=(o[0][0]+100,o[0][1]+100), pt2=(o[2][0]+100,o[2][1]+100), color=(0,255,0), thickness=5)
    # cv2.line(img, pt1=(o[3][0]+100,o[3][1]+100), pt2=(o[1][0]+100,o[1][1]+100), color=(0,255,0), thickness=5)

    # cv2.line(img, pt1=(n[0][0],n[0][1]), pt2=(n[1][0],n[1][1]), color=(0,0, 255), thickness=2)
    # cv2.line(img, pt1=(n[2][0],n[2][1]), pt2=(n[3][0],n[3][1]), color=(0,0, 255), thickness=2)
    # cv2.line(img, pt1=(n[0][0],n[0][1]), pt2=(n[2][0],n[2][1]), color=(0,0, 255), thickness=2)
    # cv2.line(img, pt1=(n[3][0],n[3][1]), pt2=(n[1][0],n[1][1]), color=(0,0, 255), thickness=2)
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
    H, HL = CalculatePerspectiveTransform(o,n)
    sw = warpTwoImages(piece,piece_t,H)
    cv2.imshow("warp", sw)
    cv2.waitKey()


if __name__ == "__main__":
    for i in range(10):
        testGenerateRandomPerspectiveTransform()

