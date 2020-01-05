import numpy as np
from skimage import data, io, util
import matplotlib.pyplot as plt
import cv2
import numpy as np

err_for_del = 0
N = np.zeros((4 * 256,), dtype=np.int32)
S = np.zeros((4 * 256,), dtype=np.int32)
out = None
B = None
contex = None
def get(im, i, j):
    if 0 <= i < im.shape[0] and 0 <= j < im.shape[1]:
        return int(im[i, j])
    return 0

def GAP(im, i, j, thre):
    # predicting context
    In = get(im, i, j-1)
    Iw = get(im, i-1, j)
    Ine = get(im, i+1, j-1)
    Inw = get(im, i-1, j-1)
    Inn = get(im, i, j-2)
    Iww = get(im, i-2, j)
    Inne = get(im, i+1, j-2)
    # input to GAP
    dh = abs(Iw - Iww) + abs(In - Inw) + abs(In - Ine)
    dv = abs(Iw - Inw) + abs(In - Inn) + abs(Ine - Inne)
    # GAP
    if dv - dh > 80:
        ic = Iw
    elif dv - dh < -80:
        ic = In
    else:
        ic = (Iw + In) / 2 + (Ine - Inw) / 4
        if dv - dh > 32:
            ic = (ic + Iw) / 2
        elif dv - dh > 8:
            ic = (3*ic + Iw) / 4
        elif dv - dh < -32:
            ic = (ic + In) / 2
        elif dv - dh < -8:
            ic = (3*ic + In) / 4
    #
    
    # Texture Quantizer
    temp = list(map(lambda x: int(x < ic),[(2*Iw)-Iww,(2*In)-Inn,Iww,Inn,Ine,Inw,Iw,In]))
    B = temp[0] << 7 | temp[1] << 6 | temp[2] << 5 | temp[3] << 4 | temp[4] << 3 | temp[5] << 2 | temp[6] << 1 | temp[7]
    
    # Delta.
    global err_for_del
    delt = dh + dv + 2*abs(err_for_del) #Error energy estimator computation
    
    # Error Energy Quantizer
    #   Now quantize error energy estimator according to thresholds given by CALIC
    #   Into 8 partitions
    Qdel = -1
    k = 0
    while k < len(thre):
        if delt <= thre[k]:
            Qdel = k
            break
        k += 1
    if Qdel == -1:
        Qdel = 7
        
    # Context Modeling Context C
    C = B * Qdel // 2
    
    # global err
    # Update N (No of occurrences)
    global N
    global S
    N[C] += 1
    S[C] += err_for_del
    # Limit the count
    if N[C] == 255:
        N[C] = N[C] / 2
        S[C] = S[C] / 2
    
    ed = S[C] // N[C]
    Itilde = ic + ed
    out[i, j] = Itilde
    context[i, j] = C  # store the context
    err_for_del = get(im, i, j) - Itilde

def calic(image, fileName):
    # d = data.coins()
    # d = io.imread("./Input_Images/mouse.png")
    d = image
    d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    thre = [5, 15, 25, 42, 60, 85, 140]
    global N
    global S
    global err_for_del
    global out
    global B
    global context
    out = np.empty((d.shape[0], d.shape[1]), dtype=np.uint8) # GAP based image
    B = np.empty((d.shape[0], d.shape[1]), dtype=np.int16) # predicted context
    err_for_del = 0
    context = np.empty((d.shape[0], d.shape[1]), dtype=np.uint8) # Final context formation
    # Final context formation given by "context" variable
    # Apply GAP to every pixel
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            GAP(d, i, j, thre)
    cv2.imwrite(fileName, out)
    # Store the un-coded calic raw image. XXX.craw
    # f = open(dst1, "wb")
    # f.write(N.tobytes())
    # f.write(S.tobytes())
    # f.write(context.tobytes())
    # f.close()
    
    # Store the raw image  XXX.raw
    # f = open(dst2, "wb")
    # f.write(d.tobytes())
    # f.close()
    
    # Apply arithmetic coding to compress it
    # print(dst1[-5:])
    # if dst1[-5:] == ".craw":
    #     dst3 = dst1.split(".")[0] + ".calic"
    # import adaptive_arithmetic_compress
    # adaptive_arithmetic_compress.main([dst1, dst3])
    
    
