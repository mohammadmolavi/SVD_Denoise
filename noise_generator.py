import numpy as np
import matplotlib.pyplot as plt

def generate(img):
    mean = 0
    sigma = 20
    gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1]))  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    if np.min(noisy_image) < 0:
        noisy_image += abs(np.min(noisy_image))
    noisy_image = noisy_image.astype(np.float32) / np.max(noisy_image)
    noisy_image *= 255
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

