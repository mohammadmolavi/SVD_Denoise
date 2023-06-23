import os
import matplotlib.pyplot as plt
from noise_generator import generate
from denoise import denoise

datadir = 'files/'
images = []
noisy = []
no_noise = []
for img in os.listdir(datadir):
    images.append(plt.imread(os.path.join(datadir, img)))
    noisy.append(generate(images[-1]))
    plt.imsave(f'./noisy/{len(noisy)}.png', noisy[-1])
    if(len(noisy) > 0):
        no_noise.append(denoise(plt.imread(f'./noisy/{len(noisy)}.png')))
    if(len(no_noise)>0):
        try:
            plt.imsave(f'./no_noise/{len(no_noise)}.jpg', no_noise[-1])
        except:
            pass
    if len(no_noise) == 9:
        for i in range(1 , 8 , 3):
            plt.subplot(331)
            plt.imshow(images[-i])
            plt.subplot(332)
            plt.imshow(noisy[-i])
            plt.subplot(333)
            plt.imshow(no_noise[-i])
            plt.subplot(334)
            plt.imshow(images[-i-1])
            plt.subplot(335)
            plt.imshow(noisy[-i-1])
            plt.subplot(336)
            plt.imshow(no_noise[-i-1])
            plt.subplot(337)
            plt.imshow(images[-i-2])
            plt.subplot(338)
            plt.imshow(noisy[-i-2])
            plt.subplot(339)
            plt.imshow(no_noise[-i-2])
            plt.show()

