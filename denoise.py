import matplotlib.pyplot as plt
import numpy as np


def svd(X):
    U, Sigma, V_transpose = np.linalg.svd(X)
    return np.dot(np.dot(U[:,:17], np.diag(Sigma[:17])), V_transpose[:17,:])

def denoise(img):
    R = svd(img[:, :, 0])
    G = svd(img[:, :, 1])
    B = svd(img[:, :, 2])
    final = np.stack((R, G, B) , axis=2)
    return final
