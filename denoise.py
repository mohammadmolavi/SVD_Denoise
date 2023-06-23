import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import time


def take_first_k(v,k):
    w = np.copy(v)
    w[k:]=0
    return(w)

def svd(X):
    U, Sigma, V_transpose = np.linalg.svd(X)
    T = take_first_k(Sigma, 35)
    return U @ np.diag(T) @ V_transpose

def denoise(img):
    img= resize(img,(170,170))
    R = svd(img[:, :, 0])
    G = svd(img[:, :, 1])
    B = svd(img[:, :, 2])
    final = np.stack((R, G, B) , axis=2)
    return final
